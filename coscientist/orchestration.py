"""Durable, code-enforced Supervisor for the scientific workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from .agents import (
    SPECIALISTS,
    SPECIALISTS_BY_STAGE,
    DeterministicProvider,
    Provider,
    bind_provider_model,
)
from .collaboration import LocalA2ATaskBus
from .disciplines import classify_discipline
from .dossier import compile_dossier
from .evidence import (
    MAX_DEEP_RESEARCH_PASSES,
    MAX_RETAINED_SOURCE_LEADS,
    SUFFICIENT_COVERAGE,
    EvidenceArtifactStore,
    EvidenceStillRunning,
    GeminiDeepResearchTransport,
    GeminiEvidenceNormalizer,
    IterativeEvidenceDiscovery,
    RegistryMetadataEnricher,
    audit_coverage,
    discovered_corpus,
    discovery_angles,
    downgrade_unlocatable_sources,
    evaluate_evidence_floor,
    merge_evidence_packets,
    merge_leads,
    names_a_document,
    resolve_manifest_locators,
    resolve_packet_locators,
    retain_leads,
    stated_identifiers,
    sweep_verification,
    unread_passes,
)
from .governance import (
    REHEARSAL_ADJUDICATOR,
    adjudicated_review_ids,
    open_blockers,
    record_adjudication,
    rehearsal_adjudications,
    withdraw_candidate,
)
from .ledger import ResearchLedger
from .methods import classify_research_mode, method_requirements
from .model_catalog import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    session_language_clause,
)
from .models import (
    EVIDENCE_FACETS,
    FACET_PHRASES,
    FORKED_STAGES,
    MAX_VERIFICATION_BATCHES,
    MERGE_PRODUCER,
    STAGES,
    VERIFICATION_BATCH_SIZE,
    ApprovalMode,
    ApprovalProfile,
    Artifact,
    ArtifactStatus,
    AuditEvent,
    Candidate,
    CandidatePopulation,
    DecisionAction,
    DeepResearchRun,
    DiscoveryCoverage,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EnrichmentRequest,
    EvidenceFloor,
    EvidencePacket,
    EvidenceRequest,
    GovernanceAdjudication,
    HumanDecision,
    KnowledgeBaseManifest,
    KnowledgeSurvey,
    ResearchPlan,
    ReviewSet,
    Session,
    SourceLead,
    utc_now,
)
from .normalization import (
    validate_candidate_comprehensiveness,
    validate_candidate_distinctness,
)
from .parity import (
    COMPARISON_CRITERIA,
    DIVERSITY_DIMENSIONS,
    detect_input_requirements,
    unresolved_blockers,
)
from .survey import write_knowledge_survey

logger = logging.getLogger(__name__)

WORKFLOW_STAGES_V1 = tuple(
    stage for stage in STAGES if stage not in {"evidence", "report"}
)
WORKFLOW_STAGES = tuple(stage for stage in STAGES if stage != "report")
MILESTONE_STAGES = frozenset({"scope", "rank", "evolve", "meta_review"})

MAX_GAP_SEARCHES = 6
"""How many searches one revision of the evidence base may dispatch.

Enough for the researcher's own request plus every facet the audit can name,
which is seven, minus the ones it usually names at once. The bound exists
because a revision is a button a researcher can press repeatedly and each press
fans out concurrently against a shared model quota; without it, five impatient
revisions are thirty-five simultaneous searches. What falls outside it is said
in the manifest rather than dropped in silence.
"""


def _deep_research_enabled() -> bool:
    """Whether this process may start a live Deep Research interaction.

    On by default. It was opt-in, and the reasoning was sound as far as it went
    -- a pass costs roughly three dollars, runs for minutes, and cannot be
    cancelled, because Vertex answers ``interactions.cancel()`` with 501
    UNIMPLEMENTED and refuses to delete an unfinished one -- but the cost of the
    default was paid on every run instead of on the accidental ones. Deep
    Research is the discovery method this system is built around; a workspace
    that quietly substitutes grounded search for it is not the system anyone
    asked for.

    What replaced the default as the guard is a ceiling the code enforces rather
    than a switch somebody remembers: :data:`coscientist.models.MAX_DISCOVERY_PASSES`
    passes and :data:`coscientist.evidence.DEFAULT_COST_LIMIT_USD` per run, both
    checked before an interaction is started. ``COSCIENTIST_DEEP_RESEARCH=off``
    still turns it off for tests, CI, and any deployment that should not spend.

    Only an explicit off turns it off, where the switch used to accept only an
    explicit on. Both readings mistrust typos, and the question is which way a
    typo should fail. It used to fail into grounded search, which costs nothing
    and is invisible in the output; now it fails into Deep Research, which is
    visible in the panel and bounded by the ceiling.
    """
    return os.environ.get("COSCIENTIST_DEEP_RESEARCH", "on").lower() not in {
        "off",
        "false",
        "0",
        "no",
    }


def _uniquely_identified(
    candidates: list[Candidate], agent: str, taken: set[str]
) -> list[Candidate]:
    """Rename candidates whose ids another generator has already handed out.

    The four generation strategies are prompted separately and each numbers its
    own output from one, so two of them return a ``cand_1`` for two unrelated
    ideas. Nothing downstream told those two apart: the tournament matched a
    candidate against itself three times out of eighteen, the ranking folded
    eight ideas into six rows, and the shortlist printed ``cand_2`` twice under a
    heading that said four candidates. Seen on a live production run.

    A colliding id is namespaced to the strategy that wrote it, which is the only
    thing distinguishing the two. Ids that do not collide are left alone, so the
    ordinary run still reads ``cand_1``.
    """
    tag = agent.removeprefix("generation").strip("_") or "generation"
    renamed: list[Candidate] = []
    for candidate in candidates:
        identifier = candidate.id
        attempt = 1
        while identifier in taken:
            attempt += 1
            suffix = tag if attempt == 2 else f"{tag}_{attempt}"
            identifier = f"{candidate.id}_{suffix}"
        taken.add(identifier)
        renamed.append(
            candidate
            if identifier == candidate.id
            else candidate.model_copy(update={"id": identifier})
        )
    return renamed


class CoScientistWorkflow:
    """Supervisor state machine.

    Specialist completion creates a draft. Advancement is a separate,
    validated transition controlled by the configured approval policy.
    """

    def __init__(
        self,
        question: str,
        provider: Provider | None = None,
        session: Session | None = None,
        *,
        approval_mode: ApprovalMode | str | None = None,
        approval_profile: ApprovalProfile | str | None = None,
        research_mode: str | None = None,
        model: str | None = None,
        language: str | None = None,
        workflow_version: int = 2,
        evidence_review: bool = False,
        rehearsal: bool = False,
        ledger: ResearchLedger | None = None,
        evidence_discovery: IterativeEvidenceDiscovery | None = None,
    ):
        self.provider = provider or DeterministicProvider()
        self.ledger = ledger
        self.evidence_discovery = evidence_discovery
        # What the page says while a stage is inside one long call. The API layer
        # attaches a writer here; left unset -- a test, the CLI -- the run says
        # nothing and behaves identically. See ``_note``.
        self.progress: Callable[[str], None] | None = None
        if session is None:
            if approval_profile is not None:
                resolved_profile = ApprovalProfile(approval_profile)
            elif approval_mode is not None:
                resolved_profile = (
                    ApprovalProfile.AUTO
                    if ApprovalMode(approval_mode) == ApprovalMode.AUTO
                    else ApprovalProfile.STAGE
                )
            else:
                resolved_profile = ApprovalProfile.MILESTONE
            resolved_mode = (
                ApprovalMode.AUTO
                if resolved_profile == ApprovalProfile.AUTO
                else ApprovalMode.HUMAN
            )
            self.session = Session(
                question=question.strip(),
                approval_mode=resolved_mode,
                approval_profile=resolved_profile,
                research_mode=research_mode or classify_research_mode(question),
                model=model or DEFAULT_MODEL,
                language=language or DEFAULT_LANGUAGE,
                discipline=classify_discipline(question),
                workflow_version=workflow_version,
                # An unattended run has nobody standing at the extra gate, and
                # ``run_auto`` accepts every draft it drafts. Recording the flag
                # as asked for would put a promise in the session that the run
                # is built never to keep, so the profile settles it here and the
                # field says what this run will actually do.
                evidence_review=evidence_review
                and resolved_profile != ApprovalProfile.AUTO,
                rehearsal=rehearsal,
                input_requirements=detect_input_requirements(question),
            )
        else:
            self.session = session
            # A resumed run keeps the model and language it was configured with.
            # Overriding them here would let a caller finish on one model a run
            # that was three stages into another, and the dossier would report a
            # single configuration for both halves.
            if model is not None and model != self.session.model:
                raise ValueError(
                    f"This session already runs on {self.session.model}; it "
                    f"cannot be resumed on {model}."
                )
            if language is not None and language != self.session.language:
                raise ValueError(
                    f"This session already reports in {self.session.language}; "
                    f"it cannot be resumed in {language}."
                )
            # A discipline, unlike a model, is a reading of the question rather
            # than a choice the caller made, so a session that never got one --
            # or got the catch-all -- is reclassified on resume.
            if (
                getattr(self.session, "discipline", "general_interdisciplinary")
                == "general_interdisciplinary"
            ):
                classified = classify_discipline(self.session.question)
                if classified != "general_interdisciplinary":
                    self.session.discipline = classified
        if not self.session.question:
            raise ValueError("A research question is required.")
        method_requirements(self.session.research_mode)
        # The session is the authority on which model this run uses. A caller
        # resuming one has to construct a provider before it can read the
        # session that answers the question, so the answer is applied here
        # rather than left to every call site to remember.
        bind_provider_model(self.provider, self.session.model)
        # The budget carried a concurrency bound that nothing read, so the bus
        # kept its own default of four however the session was configured and
        # raising the setting changed nothing about how a stage fanned out.
        self.task_bus = LocalA2ATaskBus(
            SPECIALISTS,
            self.provider,
            max_concurrency=self.session.budget.max_concurrency,
        )
        if session is None:
            event = self._event(
                "session_created",
                "supervisor",
                payload={
                    "approval_mode": self.session.approval_mode,
                    "approval_profile": self.session.approval_profile,
                    "input_requirement_ids": [
                        requirement.id
                        for requirement in self.session.input_requirements
                    ],
                },
            )
            self._persist(event)

    def seed_evidence_from(self, source: Session) -> None:
        """Take an earlier run's scope and evidence base instead of searching again.

        Gathering the corpus is the long half of a run -- eight Deep Research passes
        against one question, an hour and twenty-four dollars -- and a second run of
        the same question buys them again to arrive at the same place. A fork starts
        at generation on the corpus the first run built.

        The question has to be the one that corpus was searched against. Coverage is
        scored per research direction and per facet, both keyed on that question, so
        a corpus carried to a different one would publish a coverage figure for gaps
        nobody ever searched for. The language has to match for the same reason the
        model need not: the scope prose travels, the reasoning over it does not.

        The whole evidence half of the record travels, superseded drafts included.
        They are how that corpus came to be -- the passes, the revisions, the searches
        that returned nothing -- and a fork that carried only the final artifact would
        present a corpus nobody had ever revised. What does not travel is the
        decisions: nobody in this run approved that evidence, and the audit trail is
        not the place to invent a person who did.
        """
        if self.session.artifacts or self.session.current_stage:
            raise ValueError("Only a run that has not started can be seeded.")
        if "evidence" not in self.workflow_stages:
            raise ValueError("This workflow version has no evidence stage to seed.")
        if source.question.strip() != self.session.question:
            raise ValueError(
                f"The evidence base of {source.id} was searched against a different "
                "question, and its coverage is scored against that one. A fork keeps "
                "the question."
            )
        if source.language != self.session.language:
            raise ValueError(
                f"The scope of {source.id} is written in {source.language}; this run "
                f"reports in {self.session.language}."
            )
        carried = [
            artifact.model_copy(deep=True)
            for artifact in source.artifacts
            if artifact.stage in FORKED_STAGES
        ]
        if not any(
            artifact.stage == "evidence"
            and artifact.schema_name == "EvidencePacket"
            and artifact.status == ArtifactStatus.ACCEPTED
            for artifact in carried
        ) or not any(
            artifact.stage == "scope"
            and artifact.schema_name == "ResearchPlan"
            and artifact.status == ArtifactStatus.ACCEPTED
            for artifact in carried
        ):
            raise ValueError(
                f"{source.id} has no accepted evidence base to fork: a run has to "
                "have cleared its scope and evidence stages before another can start "
                "from them."
            )
        self.session.artifacts = carried
        self.session.seeded_evidence_from = source.id
        # The gate this flag asks for stands in the evidence stage, which a fork
        # starts past. Left set, the session recorded a stop the run was never
        # going to make, and the launcher's own summary of the run said so.
        self.session.evidence_review = False
        self.session.current_stage = self.workflow_stages.index("generate")
        self._persist(
            self._event(
                "evidence_seeded",
                "supervisor",
                payload={
                    "source_session_id": source.id,
                    "artifact_ids": [artifact.id for artifact in carried],
                },
                stage="evidence",
            )
        )

    @property
    def done(self) -> bool:
        return self.session.current_stage >= len(self.workflow_stages)

    @property
    def workflow_stages(self) -> tuple[str, ...]:
        return (
            WORKFLOW_STAGES
            if self.session.workflow_version >= 2
            else WORKFLOW_STAGES_V1
        )

    @property
    def stage(self) -> str:
        return (
            "report" if self.done else self.workflow_stages[self.session.current_stage]
        )

    @property
    def approval_mode(self) -> ApprovalMode:
        return ApprovalMode(self.session.approval_mode)

    @property
    def approval_profile(self) -> ApprovalProfile:
        return ApprovalProfile(self.session.approval_profile)

    @property
    def requires_human_approval(self) -> bool:
        if self.approval_profile == ApprovalProfile.AUTO:
            return False
        if self.approval_profile == ApprovalProfile.MILESTONE:
            # Evidence is not a milestone: the profile treats discovery as
            # internal work and hands the researcher the hypotheses built on it.
            # A run that opted in at launch stops here anyway, because a corpus
            # that came back thin looks exactly like one that came back sound
            # until somebody reads it, and by the generate gate four strategies
            # have already reasoned over whatever it was.
            return self.stage in MILESTONE_STAGES or (
                self.stage == "evidence" and self.session.evidence_review
            )
        return True

    @property
    def pending_draft(self) -> Artifact | None:
        if self.done:
            return None
        return next(
            (
                artifact
                for artifact in reversed(self.session.artifacts)
                if artifact.stage == self.stage
                and artifact.artifact_type == "stage_bundle"
                and artifact.status == ArtifactStatus.DRAFT
            ),
            None,
        )

    @property
    def pending_artifact_reviews(self) -> list[Artifact]:
        if self.approval_profile != ApprovalProfile.ARTIFACT or self.done:
            return []
        decided_ids = {
            decision.artifact_id
            for decision in self.session.decisions
            if decision.action == DecisionAction.ACCEPT
        }
        return [
            artifact
            for artifact in self.session.artifacts
            if artifact.stage == self.stage
            and artifact.artifact_type == "specialist_output"
            and artifact.status == ArtifactStatus.DRAFT
            and artifact.id not in decided_ids
        ]

    @property
    def agent_cards(self):
        return self.task_bus.agent_cards

    def _note(self, detail: str) -> None:
        """Say what this stage is doing, while it is still doing it.

        A stage is one call, and the line the workspace shows is written by the
        caller on either side of that call. Discovery escapes this because every
        Deep Research poll returns through the caller and rewrites the line on
        the way past; verification does not. A live run opened fifty-six sources
        after its eighth pass, and for the twenty-five minutes that took, the
        page held the sentence the last poll had left -- "Deep Research is still
        running; next status check in 60 seconds" -- over a stage that had
        finished searching. Nothing was wrong with the run, and nothing on the
        screen could have told a reader that.

        Narration, and narration only. A writer that raises has failed to update
        a sentence, which is not a reason to lose an hour of research, so the
        failure is logged and the stage carries on.
        """
        if self.progress is None:
            return
        try:
            self.progress(detail)
        except Exception:
            logger.exception("Could not report progress for %s", self.session.id)

    def _event(
        self,
        event_type: str,
        actor: str,
        *,
        payload: dict | None = None,
        stage: str | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            stage=stage or self.stage,
            payload=payload or {},
        )
        self.session.events.append(event)
        # Nothing had ever written this field after the session was constructed, so a
        # dossier's "Last updated" was its start time to the microsecond -- on a run
        # that took twelve minutes -- and anyone sorting saved sessions by it was
        # sorting them by when they began.
        self.session.updated_at = event.created_at
        return event

    def _persist(self, event: AuditEvent | None = None) -> None:
        if self.ledger is None:
            return
        self.ledger.save(self.session, event=event)

    async def apreview(self, feedback: str = "") -> Artifact:
        if self.done:
            return Artifact(
                stage="report",
                agent="reporter",
                content=self.render_report(),
                status=ArtifactStatus.ACCEPTED,
            )
        if self.session.status != "active":
            raise ValueError(f"Session cannot advance while {self.session.status}.")
        pending = self.pending_draft
        if pending and pending.feedback == feedback:
            return pending

        stage = self.stage
        if stage == "evidence":
            return await self._preview_evidence(feedback)
        started = time.monotonic()
        if stage == "generate":
            if self.session.workflow_version == 1:
                definitions = tuple(
                    item
                    for item in SPECIALISTS_BY_STAGE["generate"]
                    if item.role == "generation"
                )
            else:
                definitions = tuple(
                    item
                    for item in SPECIALISTS_BY_STAGE["generate"]
                    if item.role != "generation"
                )
        else:
            definitions = SPECIALISTS_BY_STAGE[stage]
        earlier_bundles = [
            item
            for item in self.session.artifacts
            if item.stage == stage and item.artifact_type == "stage_bundle"
        ]
        revision = len(earlier_bundles) + 1
        results = await self.task_bus.dispatch_stage(
            self.session,
            definitions,
            feedback=feedback,
            revision=revision,
        )
        output_ids = []
        for result in results:
            if not any(task.id == result.task.id for task in self.session.tasks):
                self.session.tasks.append(result.task)
            if not any(
                item.id == result.artifact.id for item in self.session.artifacts
            ):
                self.session.artifacts.append(result.artifact)
            output_ids.append(result.artifact.id)

        if stage == "generate":
            merged = self._merged_generation_population(results)
            if merged is not None:
                self.session.artifacts.append(merged)
                output_ids.append(merged.id)

        sections = [
            f"### {result.artifact.agent.replace('_', ' ').title()}\n\n"
            f"{result.artifact.content}"
            for result in results
        ]
        parent = earlier_bundles[-1].id if earlier_bundles else None
        draft = Artifact(
            stage=stage,
            agent="supervisor",
            content="\n\n".join(sections),
            feedback=feedback,
            version=revision,
            parent_id=parent,
            input_artifact_ids=output_ids,
        )
        self.session.artifacts.append(draft)
        event = self._event(
            "stage_drafted",
            "supervisor",
            payload={
                "artifact_id": draft.id,
                "task_ids": [result.task.id for result in results],
                # Which stage a run spends its hour in was answerable only by
                # subtracting neighbouring rows out of ``audit_events`` by hand,
                # and that reads the gap between two stages rather than the work
                # inside one -- a run parked at a gate overnight looks like a
                # stage that took all night.
                "seconds": round(time.monotonic() - started, 3),
            },
        )
        self._persist(event)
        return draft

    def _merged_generation_population(self, results) -> Artifact | None:
        """Fold the strategy generators' populations into the one the run ranks.

        Duplicates are dropped on claim and rationale, ids are made unique across
        the strategies, and the field is then held
        to ``budget.max_candidates``. The ceiling is not decoration: the ranking
        tournament plays three Swiss rounds over the whole field before the
        top-four final, so four generators that each over-produce turn a budgeted
        eighteen-comparison tournament into a fifty-match one, and the deep-dive
        section into something nobody reads. Candidates are taken a rank at a
        time across the strategies rather than a strategy at a time, so a
        ceiling thins every strategy evenly instead of deleting the last
        generator's entire contribution.
        """
        by_strategy: list[list[Candidate]] = []
        seen: set[tuple[str, str]] = set()
        taken: set[str] = set()
        for result in results:
            artifact = result.artifact
            if artifact.schema_name != "CandidatePopulation" or not artifact.payload:
                continue
            population = CandidatePopulation.model_validate(artifact.payload)
            distinct = []
            for candidate in population.candidates:
                key = (
                    candidate.claim.strip().lower(),
                    candidate.rationale.strip().lower(),
                )
                if key not in seen:
                    seen.add(key)
                    distinct.append(candidate)
            if distinct:
                by_strategy.append(
                    _uniquely_identified(distinct, artifact.agent, taken)
                )
        if not by_strategy:
            return None

        offered = sum(len(group) for group in by_strategy)
        ceiling = self.session.budget.max_candidates
        merged = [
            group[rank]
            for rank in range(max(len(group) for group in by_strategy))
            for group in by_strategy
            if rank < len(group)
        ][:ceiling]

        # The axes come with the merge, not out of a generator. The four strategy
        # populations were folded down to their candidates and nothing else, so the
        # field the run ranks reached the tournament with no criteria on it: the
        # judge had none to read, the cover printed none, and a live report and gate
        # card both said "No cross-candidate criterion was recorded" of a run whose
        # criteria are fixed. Fixed is the point -- the report says on its cover that
        # they were settled before the ideas were written, which they cannot have
        # been if the specialists proposing the ideas wrote them.
        population = CandidatePopulation(
            candidates=merged,
            target_size=len(merged),
            comparison_criteria=list(COMPARISON_CRITERIA),
            diversity_dimensions=list(DIVERSITY_DIMENSIONS),
        )
        validate_candidate_distinctness(population)
        if self.session.workflow_version >= 2:
            validate_candidate_comprehensiveness(population)

        strategies = len(by_strategy)
        content = (
            f"Merged {len(merged)} distinct candidates from {strategies} "
            f"generation {'strategy' if strategies == 1 else 'strategies'}"
        )
        if len(merged) < offered:
            # Said out loud rather than trimmed in silence: a population that
            # reads as the generators' whole output when it is not would make
            # the ranking look exhaustive over a field it never saw.
            content += (
                f"; {offered - len(merged)} further candidates were set aside to "
                f"hold the population at the budgeted ceiling of {ceiling}"
            )
        return Artifact(
            stage="generate",
            agent="generation_aggregator",
            content=content + ".",
            artifact_type="specialist_output",
            schema_name="CandidatePopulation",
            payload=population.model_dump(mode="json"),
            # This step folds the four generators' answers together and calls no
            # model. Left at the field default it reported "deterministic-offline",
            # which the dossier prints as "a fixed template (not a model)" -- the
            # phrase reserved for a stage whose specialist failed -- one line above a
            # sentence saying no stage fell back to a template.
            producer_model=MERGE_PRODUCER,
        )

    async def _preview_evidence(self, feedback: str = "") -> Artifact:
        """Run discovery first, then hand discovered leads to verification."""
        started = time.monotonic()
        earlier = [
            item
            for item in self.session.artifacts
            if item.stage == "evidence" and item.artifact_type == "stage_bundle"
        ]
        revision = len(earlier) + 1
        scope = next(
            (
                item
                for item in reversed(self.session.artifacts)
                if item.stage == "scope"
                and item.schema_name == "ResearchPlan"
                and item.status == ArtifactStatus.ACCEPTED
            ),
            None,
        )
        if scope is None:
            raise ValueError("An accepted ResearchPlan is required before discovery.")
        plan = ResearchPlan.model_validate(scope.payload)

        # A researcher who reads the evidence base and sends it back has named
        # what is missing from a corpus that already exists. Answering that by
        # starting discovery over would re-run the whole literature search --
        # another billed, uncancellable Deep Research wave, on a deployment that
        # has one -- to be told most of what is already on the page. The named
        # gaps are searched directly instead, one grounded search each, and
        # merged into the corpus that is already there.
        standing = self._draft_discovery()
        standing_manifest = (
            DiscoveryManifest.model_validate(standing.payload)
            if standing is not None
            else None
        )
        if (
            revision > 1
            and standing is not None
            and standing_manifest is not None
            and standing_manifest.source_leads
        ):
            manifest, discovery = await self._gap_directed_search(
                plan,
                standing_manifest,
                standing,
                feedback=feedback,
                revision=revision,
            )
        else:
            manifest, discovery = await self._discovered_evidence(
                plan, feedback=feedback, revision=revision
            )

        # Split before the manifest is written out, not at dispatch: the split is
        # what records how many leads went to a verifier and how many the ceiling
        # left behind, and computing it afterwards published a manifest and a
        # summary that both said nothing about either.
        batches = self._verification_batches(manifest)
        stated = self._record_discovered_corpus(manifest, feedback=feedback)
        stated_corpus = (
            EvidencePacket.model_validate(stated.payload)
            if stated is not None
            else None
        )

        summary = self._evidence_summary(manifest)
        discovery.revise(summary, manifest.model_dump(mode="json"))
        output_ids = [discovery.id]

        # Verification sees the immutable discovery manifest, but discovery itself
        # remains a draft until the complete Evidence stage is promoted.
        temporary = self.session.model_copy(deep=True)
        temp_discovery = next(
            item for item in temporary.artifacts if item.id == discovery.id
        )
        temp_discovery.status = ArtifactStatus.ACCEPTED
        # The manifest names the sources; the corpus carries what discovery said
        # about them. A verifier shown only the manifest has titles and URLs and
        # no claim to check, which is how a stage that discovered fifty-three
        # leads handed on a packet of five.
        for item in temporary.artifacts:
            if (
                item.stage == "evidence"
                and item.agent == "evidence_discovery"
                and item.schema_name == "EvidencePacket"
                and item.status == ArtifactStatus.DRAFT
            ):
                item.status = ArtifactStatus.ACCEPTED
        # Only the ones nothing has answered yet. Left unfiltered, this re-ran
        # every gap search the previous revision had already completed, on every
        # later revision, growing by a wave each time.
        pending_enrichment = [
            request
            for request in manifest.enrichment_requests
            if request.status in {"queued", "working"}
        ]
        if pending_enrichment:
            enrichment_definition = tuple(
                item
                for item in SPECIALISTS_BY_STAGE["evidence"]
                if item.role == "evidence_discovery"
            )
            enrichment_results = await self.task_bus.dispatch_stage(
                temporary,
                enrichment_definition,
                feedback=(
                    "Resolve only these residual searches (maximum six):\n"
                    + "\n".join(
                        f"- {request.query}" for request in pending_enrichment[:6]
                    )
                ),
                revision=revision,
            )
            for result in enrichment_results:
                result.artifact.status = ArtifactStatus.DRAFT
                temporary.tasks.append(result.task)
                temporary.artifacts.append(result.artifact.model_copy(deep=True))
                temporary.artifacts[-1].status = ArtifactStatus.ACCEPTED
                self.session.tasks.append(result.task)
                self.session.artifacts.append(result.artifact)
                output_ids.append(result.artifact.id)

        verifier_definition = tuple(
            item
            for item in SPECIALISTS_BY_STAGE["evidence"]
            if item.role == "source_verification"
        )
        # Counted as they land rather than at the end, because the end is the
        # thing that is far away: the sources are opened one at a time inside a
        # batch, the batches run together, and the whole fan-out is the longest
        # silence in the run. A reader watching it is owed the count.
        opening = sum(len(batch) for batch in batches)
        noun = "source" if opening == 1 else "sources"
        self._note(f"Opening {opening} {noun} to check what the documents say.")
        checked = 0

        async def verified(batch: list[SourceLead], index: int):
            group = await self.task_bus.dispatch_stage(
                temporary,
                verifier_definition,
                feedback=self._verification_feedback(
                    feedback, batch, index, len(batches), stated_corpus
                ),
                revision=revision,
            )
            nonlocal checked
            checked += len(batch)
            self._note(
                f"Checked {checked} of {opening} {noun} "
                f"against what the document actually says."
            )
            return group

        dispatched = await asyncio.gather(
            *(verified(batch, index) for index, batch in enumerate(batches, start=1))
        )
        results = [result for group in dispatched for result in group]
        verified_packets: list[EvidencePacket] = []
        for result in results:
            self.session.tasks.append(result.task)
            if result.artifact.schema_name == "EvidencePacket" and (
                result.artifact.payload
            ):
                # A status is a claim about an act, and "verified" claims someone
                # opened the document. Nobody can have opened a bare domain, so
                # the assertion is corrected here rather than trusted: the report
                # downstream reads the status and nothing else.
                checked = downgrade_unlocatable_sources(
                    EvidencePacket.model_validate(result.artifact.payload)
                )
                # And then every remaining locator is actually fetched. Whether
                # the specialist called its tool, and what the tool returned, are
                # facts rather than judgements, so they are established here
                # instead of being taken on trust -- a run whose only retrieval
                # tool raised ImportError on every call still returned a packet
                # full of confident statuses.
                checked = await self._swept(checked)
                verified_packets.append(checked)
                result.artifact.payload = checked.model_dump(mode="json")
            self.session.artifacts.append(result.artifact)
            output_ids.append(result.artifact.id)

        # The batches are one verification of one corpus, and everything
        # downstream reads the newest EvidencePacket in the session: left as ten
        # artifacts, the gate, the panel and the list of citable ids would all
        # see whichever batch happened to finish last. They are folded into one
        # here for the same reason the discovery angles are.
        #
        # The corpus discovery stated is folded in behind them. Verification
        # answers about sources, and the findings the search recorded against
        # those sources are not its to discard by omission -- a verifier that
        # returns an entry per source and no claims would undo the corpus
        # written above, which is this very failure arriving one stage later.
        # The verified packets come first, so wherever verification spoke about
        # a document its status is the one that stands, and what is left over is
        # carried at the tier discovery is entitled to assert and no higher.
        folded = verified_packets + ([stated_corpus] if stated_corpus else [])
        if len(folded) > 1:
            consolidated = merge_evidence_packets(self.session.question, folded)
            for result in results:
                if result.artifact.schema_name == "EvidencePacket":
                    result.artifact.status = ArtifactStatus.SUPERSEDED
            corpus = Artifact(
                stage="evidence",
                agent="source_verification",
                artifact_type="specialist_output",
                content=(
                    f"### Verified corpus\n\n{len(consolidated.sources)} distinct "
                    f"sources and {len(consolidated.claims)} claims, merged from "
                    f"{len(verified_packets)} verification batches"
                    + (" and what discovery stated." if stated_corpus else ".")
                ),
                feedback=feedback,
                producer_model=getattr(self.provider, "model_id", "unknown"),
                schema_name="EvidencePacket",
                payload=consolidated.model_dump(mode="json"),
                input_artifact_ids=[result.artifact.id for result in results],
            )
            self.session.artifacts.append(corpus)
            output_ids.append(corpus.id)
            verified_packets = [consolidated]

        if verified_packets:
            manifest = self._manifest_with_verification(manifest, verified_packets)
            discovery.revise(
                self._evidence_summary(manifest), manifest.model_dump(mode="json")
            )

        # Here rather than at render time, for two reasons that both make it the
        # only place it can go. The report is computed on demand from the stored
        # session, so a section that needed a model call to draw would be a model
        # call inside a page request; and the reports being merged live in the
        # artifact store, which is the one thing the stored session does not
        # carry, so this is the last point in the run that can still reach them.
        survey = await asyncio.to_thread(self._knowledge_survey, manifest)
        if survey is not None:
            manifest = manifest.model_copy(update={"knowledge_survey": survey})
            discovery.revise(
                self._evidence_summary(manifest), manifest.model_dump(mode="json")
            )

        # The corpus once, never the specialist's surrounding prose. A live run
        # ended its answer by pasting the navigation menu of a university site
        # its fetch tool had returned -- "Skip to main content", the whole course
        # listing -- and that landed verbatim in the artifact a researcher reads.
        # Batched, the same habit would paste it ten times.
        if verified_packets:
            verifier_text = "### Source Verification\n\n" + json.dumps(
                verified_packets[0].model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        else:
            verifier_text = "\n\n".join(
                f"### Source Verification\n\n{result.artifact.content}"
                for result in results
            )
        draft = Artifact(
            stage="evidence",
            agent="supervisor",
            content=f"{summary}\n\n{verifier_text}".strip(),
            feedback=feedback,
            version=revision,
            parent_id=earlier[-1].id if earlier else None,
            input_artifact_ids=output_ids,
        )
        self.session.artifacts.append(draft)
        self._persist(
            self._event(
                "evidence_stage_drafted",
                "supervisor",
                payload={
                    "artifact_id": draft.id,
                    "passes": len(manifest.runs),
                    "convergence_reason": manifest.convergence_reason,
                    # This stage is resumed rather than run once: a poll that
                    # raises ``EvidenceStillRunning`` never reaches here, and a
                    # worker that dies hands the lease on and starts the sweep
                    # again. So this is the last attempt's own time, not the
                    # forty to fifty minutes the stage costs end to end.
                    "seconds": round(time.monotonic() - started, 3),
                },
            )
        )
        return draft

    def _draft_discovery(self, *, require_payload: bool = True) -> Artifact | None:
        """The discovery manifest this stage is still working on, if there is one.

        Evidence is the one stage whose specialist output survives a revision:
        the manifest is revised in place across Deep Research polls, across a
        grounded fallback and across a researcher sending the corpus back, so
        every path into the stage has to find the same artifact rather than
        start a second one beside it.
        """
        return next(
            (
                item
                for item in reversed(self.session.artifacts)
                if item.stage == "evidence"
                and item.agent == "deep_research_discovery"
                and item.schema_name == "DiscoveryManifest"
                and item.status == ArtifactStatus.DRAFT
                and (item.payload or not require_payload)
            ),
            None,
        )

    async def _discovered_evidence(
        self, plan: ResearchPlan, *, feedback: str, revision: int
    ) -> tuple[DiscoveryManifest, Artifact]:
        """Search the literature from nothing, by whichever provider is wired up."""
        controller = self.evidence_discovery
        transport: GeminiDeepResearchTransport | None = None
        transport_error = ""
        if controller is None and _deep_research_enabled():
            # Deep Research runs on Vertex AI with Application Default Credentials
            # or on an API key; the transport decides which. Construction failure
            # is a degraded mode, never a crashed run, so the workflow can still
            # reach the evidence gate and report why nothing was discovered.
            try:
                transport = GeminiDeepResearchTransport()
            except Exception as exc:  # any client failure degrades, never crashes
                transport_error = str(exc)
        elif controller is None:
            transport_error = (
                "Deep Research was turned off for this deployment "
                "(COSCIENTIST_DEEP_RESEARCH=off), so the literature was searched "
                "with grounded web search instead."
            )
        if controller is None and transport is not None:
            controller = IterativeEvidenceDiscovery(
                transport,
                EvidenceArtifactStore(),
                # One interaction per evidence facet, all seven at once, then a
                # single pass at whatever they left open. The sequential
                # gap-directed loop this replaces was smarter per dollar and
                # seven times slower, and it only ever asked its second question
                # after the first had come back.
                fan_out=os.environ.get("EVIDENCE_FAN_OUT", "true").lower() == "true",
                max_waves=2,
                registry_enricher=RegistryMetadataEnricher(),
                polls_per_invocation=(
                    1
                    if os.environ.get("EVIDENCE_TASK_STEP_MODE", "false").lower()
                    == "true"
                    else None
                ),
            )
        if controller is None:
            manifest, discovery = await self._search_grounded_discovery(
                plan,
                transport_error,
                feedback=feedback,
                revision=revision,
            )
        else:
            discovery = self._draft_discovery(require_payload=False)
            existing_manifest = (
                DiscoveryManifest.model_validate(discovery.payload)
                if discovery is not None and discovery.payload
                else DiscoveryManifest(question=self.session.question)
            )
            if discovery is None:
                discovery = Artifact(
                    stage="evidence",
                    agent="deep_research_discovery",
                    artifact_type="specialist_output",
                    content="Deep Research pass 1 of 3 is being started.",
                    feedback=feedback,
                    producer_model="deep-research-preview-04-2026",
                    schema_name="DiscoveryManifest",
                    payload=existing_manifest.model_dump(mode="json"),
                )
                self.session.artifacts.append(discovery)
                self._persist()

            def persist_manifest(updated: DiscoveryManifest) -> None:
                discovery.revise(
                    self._evidence_summary(updated), updated.model_dump(mode="json")
                )
                self._persist()

            normalizer = None
            if isinstance(controller.transport, GeminiDeepResearchTransport):
                try:
                    normalizer = GeminiEvidenceNormalizer()
                except Exception:
                    # Deterministic paragraph extraction is the documented
                    # fallback; losing the model normalizer must not discard a
                    # report that Deep Research already paid to produce.
                    normalizer = None

            manifest = await asyncio.to_thread(
                controller.run,
                self.session,
                plan,
                manifest=existing_manifest,
                normalizer=normalizer,
                manifest_callback=persist_manifest,
            )
            if any(
                run.status in {"queued", "in_progress", "requires_action"}
                for run in manifest.runs
            ):
                raise EvidenceStillRunning(
                    "Deep Research interaction is still running."
                )
            # Deep Research names its sources with the same grounding redirector
            # search uses. They are followed here, before the manifest is
            # recorded, so the corpus written from these leads names documents.
            manifest = await self._resolved_lead_locators(manifest)
        return manifest, discovery

    def _searched_facets(self, manifest: DiscoveryManifest) -> set[str]:
        """Which facets this corpus has already been searched under.

        Coverage distinguishes a facet a pass was aimed at and came back empty
        from one nobody asked about, and only the caller knows which is which.
        On a second visit to the stage the original caller is gone, so it is
        reconstructed from the three places the act was recorded: the angles a
        grounded pass ran, the facet each Deep Research pass was sent to cover,
        and the facet tags on the leads that came back.
        """
        return {
            *(angle for angle in manifest.discovery_angles if angle in EVIDENCE_FACETS),
            *(run.facet for run in manifest.runs if run.facet in EVIDENCE_FACETS),
            *(
                facet
                for lead in manifest.source_leads
                for facet in lead.facets
                if facet in EVIDENCE_FACETS
            ),
        }

    def _discovered_corpus(self) -> Artifact | None:
        """The newest packet discovery has written, merged or otherwise."""
        return next(
            (
                item
                for item in reversed(self.session.artifacts)
                if item.stage == "evidence"
                and item.agent == "evidence_discovery"
                and item.schema_name == "EvidencePacket"
                and item.payload
            ),
            None,
        )

    @staticmethod
    def _corpus_statements(packet: Artifact | None) -> list[DiscoveryStatement]:
        """Recover what discovery said, from the corpus rather than the manifest.

        Coverage scores its facets from leads but its research directions from
        statements, and statements are the one thing the manifest does not keep
        on the grounded path -- they are computed at dispatch and dropped. Left
        unrecovered, a gap pass would re-score every direction against only the
        handful of statements its own searches returned, and a revision that
        added sources would report coverage falling.

        The merged corpus packet holds them: its sources carry the facet of the
        search that found them, and its claims name their source.
        """
        if packet is None:
            return []
        corpus = EvidencePacket.model_validate(packet.payload)
        by_id = {source.id: source for source in corpus.sources}
        return [
            DiscoveryStatement(
                text=claim.claim,
                facet=source.facet if source.facet in EVIDENCE_FACETS else "",
                source_urls=[source.url],
                originating_pass=1,
                relation=claim.relation,
            )
            for claim in corpus.claims
            if (source := by_id.get(claim.source_id or "")) is not None
        ]

    @staticmethod
    def _never_less_covered(
        audited: DiscoveryCoverage, previous: DiscoveryCoverage | None
    ) -> DiscoveryCoverage:
        """Hold each score at the best this corpus has ever measured.

        Nothing is ever removed from the evidence base, so no direction and no
        facet can become less covered by searching for more. The audit can still
        say it did: a direction is scored on how many statements exist, and the
        statements a revision re-derives come from the merged corpus, where two
        passes that found the same paper for the same finding have become one.
        The first wave's seven near-identical statements come back as one, and a
        revision that added a retraction notice reported coverage falling from
        88% to 63% -- which a researcher reads as "asking for more made it
        worse", about the button they just pressed.
        """
        if previous is None:
            return audited
        floored = audited.model_copy(deep=True)
        floored.direction_scores = {
            direction: max(score, previous.direction_scores.get(direction, 0.0))
            for direction, score in audited.direction_scores.items()
        }
        floored.facet_scores = {
            facet: max(score, previous.facet_scores.get(facet, 0.0))
            for facet, score in audited.facet_scores.items()
        }
        totals = [*floored.facet_scores.values(), *floored.direction_scores.values()]
        floored.weighted_score = round(sum(totals) / max(1, len(totals)), 4)
        floored.sufficient = (
            floored.weighted_score >= SUFFICIENT_COVERAGE and not floored.gaps
        )
        return floored

    async def _gap_directed_search(
        self,
        plan: ResearchPlan,
        manifest: DiscoveryManifest,
        discovery: Artifact,
        *,
        feedback: str,
        revision: int,
    ) -> tuple[DiscoveryManifest, Artifact]:
        """Search the named gaps, and leave the rest of the corpus alone.

        This is what a revision at the evidence gate does. The alternative --
        starting the discovery wave again -- costs another seven Deep Research
        interactions, roughly twenty-one dollars, forty minutes, and cannot be
        cancelled once begun, to re-find the sources already on the page. What
        the researcher asked for is the missing part, so only that is searched,
        with the same grounded-search specialist the fallback path uses.

        Each target becomes one search and one ``EnrichmentRequest``. It is not
        recorded as a ``DeepResearchRun`` because it is not one: the manifest's
        run list is capped at the Deep Research ceiling and the panel counts it
        as paid passes, so filing gap searches there would both spend the cap
        and tell the reader money was spent that was not.
        """
        # Read before anything is dispatched. The gap searches append their own
        # packets to the session, so looked up afterwards "the corpus so far"
        # would be whichever gap search happened to land last.
        prior = self._discovered_corpus()
        prior_statements = self._corpus_statements(prior)
        latest = manifest.coverage_history[-1] if manifest.coverage_history else None
        impact_rank = {"blocking": 0, "high": 1, "medium": 2, "low": 3}
        open_gaps = sorted(
            (gap for gap in (latest.gaps if latest else []) if gap.status == "open"),
            key=lambda gap: (impact_rank.get(gap.decision_impact, 2), -gap.priority),
        )
        targets: list[tuple[str, str, list[str]]] = []
        if feedback.strip():
            # First, and its own target: the researcher read the corpus and said
            # what is wrong with it, which is better aimed than anything the
            # coverage audit can infer. Folded into the gap prompts instead, it
            # would be repeated seven times and answered once.
            targets.append(
                (
                    "",
                    "The researcher reviewed this evidence base and asked for "
                    f"this specifically: {feedback.strip()}",
                    [],
                )
            )
        targets.extend(
            (
                gap.facet if gap.facet in EVIDENCE_FACETS else "",
                gap.description,
                [gap.id],
            )
            for gap in open_gaps
        )
        if not targets:
            # Nothing named a gap and nobody said why they sent it back, so the
            # weakest facets are searched. Sending it back has to do something.
            weakest = sorted(
                EVIDENCE_FACETS,
                key=lambda facet: (
                    latest.facet_scores.get(facet, 0.0) if latest else 0.0
                ),
            )[:MAX_GAP_SEARCHES]
            targets = [
                (
                    facet,
                    "Search again for "
                    f"{FACET_PHRASES.get(facet, facet.replace('_', ' '))}; this "
                    "facet is the thinnest part of the evidence base.",
                    [],
                )
                for facet in weakest
            ]
        selected = targets[:MAX_GAP_SEARCHES]
        dropped = len(targets) - len(selected)

        definition = tuple(
            item
            for item in SPECIALISTS_BY_STAGE["evidence"]
            if item.role == "evidence_discovery"
        )
        directions = "\n".join(f"- {item}" for item in plan.success_criteria)
        known = "\n".join(
            f"- {lead.title or lead.canonical_url}"
            for lead in manifest.source_leads[:40]
        )
        shared = (
            "GAP-DIRECTED PASS. A literature search has already run for this "
            "question and its results were reviewed by a researcher, who sent "
            "them back. You are filling one named hole in that corpus, not "
            "repeating the search. Return a typed EvidencePacket whose claims "
            "each name the source they came from. Everything stays "
            "discovered_unverified: you have read search results, not sources. "
            "A verification specialist runs next.\n\n"
            f"Research question: {self.session.question}\n"
            f"What the plan must be able to show:\n{directions}\n"
            f"Already in the corpus -- finding these again adds nothing:\n{known}"
        )
        dispatched = await asyncio.gather(
            *(
                self.task_bus.dispatch_stage(
                    self.session,
                    definition,
                    feedback=(
                        f"{shared}\n\nGap {index} of {len(selected)}"
                        + (f" ({facet})" if facet else "")
                        + f". {brief} Stay on this gap: the others are being "
                        "searched in parallel, and duplicating them wastes the "
                        "pass."
                    ),
                    revision=revision,
                )
                for index, (facet, brief, _) in enumerate(selected, start=1)
            )
        )
        target_by_result = {
            id(result): selected[index]
            for index, batch in enumerate(dispatched)
            for result in batch
            if index < len(selected)
        }

        packets: list[EvidencePacket] = []
        statements: list[DiscoveryStatement] = []
        leads = list(manifest.source_leads)
        requests: list[EnrichmentRequest] = []
        results = [result for batch in dispatched for result in batch]
        for result in results:
            self.session.tasks.append(result.task)
            facet, brief, gap_ids = target_by_result.get(id(result), ("", "", []))
            request = EnrichmentRequest(
                provider="google_search",
                gap_ids=list(gap_ids),
                query=brief,
                # Recorded as done here rather than left queued: the residual
                # enrichment dispatch downstream picks up anything still pending,
                # and it would run these searches a second time.
                status="completed" if result.artifact.payload else "failed",
                result_artifact_reference=result.artifact.id,
            )
            requests.append(request)
            if not result.artifact.payload:
                self.session.artifacts.append(result.artifact)
                continue
            packet = await self._resolved_locators(
                EvidencePacket.model_validate(result.artifact.payload)
            )
            relations_by_source: dict[str, list[str]] = {}
            for claim in packet.claims:
                if claim.source_id:
                    relations_by_source.setdefault(claim.source_id, []).append(
                        claim.relation
                    )
            if facet:
                for source in packet.sources:
                    source.facet = facet
            # Every search contributes its statements, whether or not it was
            # aimed at one of the seven facets. Only the facet tag depends on
            # that. A revision's own target -- what the researcher wrote in the
            # box -- carries no facet, and skipping it here meant the common
            # revision, the one where the audit named no gaps and the request is
            # the only search, contributed no statements at all. Coverage scores
            # research directions by counting statements, so a live revision
            # scored every direction zero and the panel reported coverage
            # falling from 88% to 70% for adding sources to the corpus.
            statements.extend(
                DiscoveryStatement(
                    text=claim.claim,
                    facet=facet,
                    source_urls=[
                        source.url
                        for source in packet.sources
                        if source.id == claim.source_id
                    ],
                    originating_pass=1,
                    relation=claim.relation,
                )
                for claim in packet.claims
            )
            result.artifact.payload = packet.model_dump(mode="json")
            self.session.artifacts.append(result.artifact)
            packets.append(packet)
            leads = merge_leads(
                leads,
                [
                    SourceLead(
                        canonical_url=source.url,
                        title=source.title,
                        source_type=source.source_type,
                        provider="google_search",
                        # What the locator states, so a paper the corpus already
                        # holds under its DOI is recognised as that paper rather
                        # than added again under its address.
                        identifiers={
                            **stated_identifiers(source.url),
                            **source.identifiers,
                        },
                        originating_passes=[1],
                        originating_statement_ids=list(source.supports_claim_ids),
                        facets=[facet] if facet else [],
                        claim_relations=list(
                            dict.fromkeys(relations_by_source.get(source.id, []))
                        ),
                        raw_artifact_reference=result.artifact.id,
                    )
                    for source in packet.sources
                ],
            )

        # Rebuilt from the corpus that already existed plus what the gap searches
        # returned. Merging only the new packets would hand verification a corpus
        # containing nothing but the gap material, and everything downstream
        # reads the newest packet in the session.
        new_packets = len(packets)
        if prior is not None:
            packets.insert(0, EvidencePacket.model_validate(prior.payload))
        merged = self._merged_discovery_corpus(packets, results, feedback=feedback)
        if merged is not None and prior is not None:
            # Superseded by hand: the merge only supersedes the artifacts it was
            # handed as results, and the previous revision's corpus is not one of
            # them. Left a draft, two packets would each claim to be the corpus
            # and everything downstream reads whichever is newest.
            prior.status = ArtifactStatus.SUPERSEDED

        searched = self._searched_facets(manifest) | {
            facet for facet, _, _ in selected if facet
        }
        updated = manifest.model_copy(deep=True)
        retained = retain_leads(leads, MAX_RETAINED_SOURCE_LEADS)
        updated.leads_beyond_retention_ceiling += len(leads) - len(retained)
        updated.source_leads = retained
        updated.verification_handoff_source_ids = [lead.id for lead in retained]
        updated.enrichment_requests = [*updated.enrichment_requests, *requests]
        # Whatever discovery scored, scored again. The two discovery paths key
        # their directions differently -- a Deep Research narrative is keyed on
        # the question, the grounded fallback on the success criteria -- and a
        # revision that re-keyed them turned the floor below into a no-op, since
        # a direction the previous audit never named has no previous score to be
        # held at. That is how a live revision published 70% under an 88%.
        directions = list(latest.direction_scores) if latest else []
        updated.coverage_history = [
            *updated.coverage_history,
            self._never_less_covered(
                audit_coverage(
                    DiscoveryNarrative(
                        question=self.session.question,
                        research_directions=directions or list(plan.success_criteria),
                        statements=[*prior_statements, *statements],
                    ),
                    retained,
                    previous=latest,
                    searched_facets=searched,
                ),
                latest,
            ),
        ]
        found = sum(
            len(packet.sources) for packet in packets[len(packets) - new_packets :]
        )
        updated.convergence_reason = "gap_directed_search"
        # No silent caps, and the gap list alone is not enough of one: a gap that
        # was searched and came back empty stays on that list beside a gap this
        # revision never got to, and they read identically.
        updated.gap_searches_deferred = dropped
        updated.synthesis_report = (
            f"{updated.synthesis_report}\n\nRevision {revision}: "
            f"{len(selected)} gap-directed searches added {found} sources."
        ).strip()
        if dropped:
            updated.synthesis_report += (
                f" {dropped} further gaps were left open to hold this revision "
                f"at {MAX_GAP_SEARCHES} searches; they remain listed as gaps."
            )
        discovery.revise(
            self._evidence_summary(updated), updated.model_dump(mode="json")
        )
        return updated, discovery

    @staticmethod
    def _verification_batches(manifest: DiscoveryManifest) -> list[list[SourceLead]]:
        """Split the discovered leads into work-lists one specialist can finish.

        One dispatch over the whole corpus does not scale with the corpus, and
        the failure is silent: shown ninety leads, a live run returned a packet
        of five, three of which named a bare domain the model had shortened for
        itself. Nothing in the packet said the other eighty-five had been
        skipped -- they simply were not in it, and the evidence floor then
        measured the literature as one usable source.

        A bounded list is a different instruction. Each batch is small enough to
        enumerate, they run concurrently, and a batch that comes back short is
        visibly short against the leads it was given.
        """
        leads = [
            lead
            for lead in manifest.source_leads
            if names_a_document(lead.canonical_url)
        ]
        if not leads:
            # Nothing was discovered by URL, so the specialist works from the
            # corpus packet as before rather than from an empty work-list.
            manifest.leads_sent_to_verification = 0
            manifest.leads_beyond_verification_ceiling = 0
            return [[]]
        # Highest-value leads first, so a corpus past the ceiling loses its
        # weakest material rather than an arbitrary tail.
        leads.sort(
            key=lambda lead: (not lead.identifiers, lead.source_type == "unknown")
        )
        capped = leads[: VERIFICATION_BATCH_SIZE * MAX_VERIFICATION_BATCHES]
        manifest.leads_sent_to_verification = len(capped)
        manifest.leads_beyond_verification_ceiling = len(leads) - len(capped)
        return [
            capped[start : start + VERIFICATION_BATCH_SIZE]
            for start in range(0, len(capped), VERIFICATION_BATCH_SIZE)
        ]

    @staticmethod
    def _verification_feedback(
        feedback: str,
        batch: list[SourceLead],
        index: int,
        total: int,
        corpus: EvidencePacket | None = None,
    ) -> str:
        # What the search said each of these documents shows, so the verifier is
        # checking a stated finding against the text rather than reading a title
        # and writing down whatever it can make of it. Without this the batch is
        # a list of addresses, and a verifier handed addresses returns addresses:
        # a live run reached the panel with ninety-two sources and ten findings.
        findings: dict[str, list[str]] = {}
        if corpus is not None:
            by_id = {source.id: source.url for source in corpus.sources}
            for claim in corpus.claims:
                url = by_id.get(claim.source_id or "")
                if url and claim.claim:
                    findings.setdefault(url, []).append(claim.claim)
        if not batch:
            return feedback
        lines: list[str] = []
        for lead in batch:
            lines.append(
                f"- {lead.canonical_url}" + (f" -- {lead.title}" if lead.title else "")
            )
            lines.extend(
                f"    - What the search says it shows: {text}"
                for text in findings.get(lead.canonical_url, [])
            )
        listing = "\n".join(lines)
        stated = (
            " Each finding listed under a source is a claim to check against "
            "that document and carry into your packet with a status of its own: "
            "keep its wording, and where the text does not support it say so "
            "rather than dropping it."
            if findings
            else ""
        )
        return (
            f"Verify exactly these {len(batch)} sources. They are batch {index} of "
            f"{total}; the others are being verified in parallel, so do not reach "
            "for sources outside this list and do not shorten a URL to its "
            "domain -- fetch each locator exactly as written. Return one entry "
            f"per source, including the ones you could not reach.{stated}\n\n"
            f"{listing}\n\n{feedback}"
        ).strip()

    async def _swept(self, packet: EvidencePacket) -> EvidencePacket:
        """Fetch every locator in a packet, and carry on if the network will not.

        A discovery pass that found real literature must not be lost because the
        machine running it has no outbound access. What could not be reached
        keeps whatever the specialist said, and the packet's own limitations
        already record how it was reached.
        """
        try:
            return await sweep_verification(packet)
        except Exception:
            return packet

    def _knowledge_survey(self, manifest: DiscoveryManifest) -> KnowledgeSurvey | None:
        """Merge the search passes into one cited survey, or leave them as they are.

        Every failure is swallowed on purpose. What this produces is one section
        of the report, and the section exists without it: the Knowledge Base
        reproduces the pass reports where no survey was written, which is what it
        did before this stage learned to merge them. Raising here would throw
        away a completed evidence stage -- seven Deep Research passes and the
        verification behind them -- over the presentation of what it found.
        """
        try:
            return write_knowledge_survey(
                manifest,
                self.provider,
                language=session_language_clause(self.session),
            )
        except Exception:
            logger.exception(
                "The knowledge survey failed; the Knowledge Base will reproduce "
                "the %d search passes instead.",
                len(manifest.runs),
            )
            return None

    def _manifest_with_verification(
        self, manifest: DiscoveryManifest, packets: list[EvidencePacket]
    ) -> DiscoveryManifest:
        """Write each source's verification outcome back onto its discovery lead.

        The evidence panel is rendered from the manifest, and the manifest had no
        way to hold a verification result -- ``SourceLead.verification_status``
        was typed as the single literal ``discovered_unverified``. So the panel
        showed forty-four leads all marked unverified next to a verification
        stage that had run, and a reader had no way to tell which sources the
        run could actually stand on.

        Retrieval is also where a lead learns its DOI, and a DOI is what makes
        two leads one document, so the corpus is re-merged before it is handed
        back. Left as written, two rows for one paper stood in the panel until
        the next merge folded them -- and the next merge is the gap search a
        researcher asks for at this gate, so a live run answered "search for
        long-term safety" by reporting the corpus had gone from 88 leads to 85.
        Nothing had been removed. The count had been wrong when they read it.
        """
        updated = manifest.model_copy(deep=True)
        by_url = {source.url: source for packet in packets for source in packet.sources}
        relations: dict[str, list[str]] = {}
        for packet in packets:
            urls = {source.id: source.url for source in packet.sources}
            for claim in packet.claims:
                url = urls.get(claim.source_id or "")
                if url:
                    relations.setdefault(url, []).append(claim.relation)
        for lead in updated.source_leads:
            source = by_url.get(lead.canonical_url)
            if source is None:
                continue
            lead.verification_status = source.verification_status
            lead.verification_note = source.verification_note
            if source.authors:
                lead.authors = source.authors
            if source.year:
                lead.year = source.year
            if source.identifiers:
                lead.identifiers = {**source.identifiers, **lead.identifiers}
            if source.facet and source.facet not in lead.facets:
                lead.facets = [*lead.facets, source.facet]
            lead.claim_relations = list(
                dict.fromkeys(
                    [*lead.claim_relations, *relations.get(lead.canonical_url, [])]
                )
            )
        updated.source_leads = merge_leads([], updated.source_leads)
        updated.verification_handoff_source_ids = [
            lead.id for lead in updated.source_leads
        ]
        return updated

    async def _search_grounded_discovery(
        self,
        plan: ResearchPlan,
        transport_error: str,
        *,
        feedback: str,
        revision: int,
    ) -> tuple[DiscoveryManifest, Artifact]:
        """Discover sources with grounded search when Deep Research is off.

        Deep Research is billable and opt-in, so most runs never have it. The
        branch this replaces answered that by writing an empty manifest, and the
        entire evidence stage then collapsed: nothing discovered, nothing to
        verify, and -- correctly, but uselessly -- not one hypothesis downstream
        able to cite anything. The grounded-search discovery specialist is
        already built and wired for gap enrichment, and DeepMind's own system
        searches the literature rather than commissioning a research report, so
        it does the broad pass instead of leaving the stage empty.

        What it returns is weaker than a Deep Research report, and the manifest
        says so: the attempted pass is recorded with the reason it did not run,
        every lead carries ``provider="google_search"``, and nothing is verified
        here. Verification runs next, against material that now exists.

        Coverage is scored, though, which it was not. Only the Deep Research
        controller used to call :func:`audit_coverage`, so on this path the
        manifest's ``coverage_history`` stayed empty and the evidence panel
        rendered "Coverage by facet" and "Unresolved gaps" as blank boxes -- on
        the majority of runs, and with no indication that the blankness meant
        unmeasured rather than perfect. It is measured here from the angle each
        packet was searched under, which is better evidence of what a source is
        than the keyword heuristic: the search that found it is what defines it.
        """
        existing = self._draft_discovery()
        if existing is not None:
            manifest = DiscoveryManifest.model_validate(existing.payload)
            if manifest.source_leads:
                # Discovery that already happened -- a paid Deep Research pass
                # earlier in this session, or a previous revision of this stage
                # -- must not be deleted just because the switch is off now.
                return manifest, existing

        definition = tuple(
            item
            for item in SPECIALISTS_BY_STAGE["evidence"]
            if item.role == "evidence_discovery"
        )
        directions = "\n".join(f"- {item}" for item in plan.success_criteria)
        angles = discovery_angles(plan)
        shared = (
            "PRIMARY DISCOVERY PASS. Deep Research is unavailable for this "
            "session, so nothing has searched the literature yet and you are "
            "not enriching anyone's gaps. Return a typed EvidencePacket whose "
            "claims each name the source they came from. Everything stays "
            "discovered_unverified: you have read search results, not sources. "
            "A verification specialist runs next.\n\n"
            f"Research question: {self.session.question}\n"
            f"What the plan must be able to show:\n{directions}"
        )
        # One search per angle, dispatched together. The single broad query this
        # replaces asked for the mechanism, the studies for and against it,
        # replications, negative results, retractions and the measurement
        # standards all at once, and a live run answered it with four sources,
        # every one of them supporting. Asked one at a time, each of those is a
        # search that returns its own literature.
        dispatched = await asyncio.gather(
            *(
                self.task_bus.dispatch_stage(
                    self.session,
                    definition,
                    feedback=(
                        f"{shared}\n\nSearch angle {index} of {len(angles)} "
                        f"({angle.key}). {angle.brief} Stay on this angle: the "
                        "other angles of this question are being searched in "
                        "parallel, and duplicating them costs the pass its "
                        f"breadth.\n\n{feedback}"
                    ).strip(),
                    revision=revision,
                )
                for index, angle in enumerate(angles, start=1)
            )
        )
        results = [result for batch in dispatched for result in batch]
        # dispatch_stage returns one result per specialist, and there is one
        # specialist in this definition, so the batches line up with the angles
        # that produced them. That correspondence is what lets a facet be
        # recorded rather than inferred.
        angle_by_result = {
            id(result): angles[index]
            for index, batch in enumerate(dispatched)
            for result in batch
            if index < len(angles)
        }
        packets: list[EvidencePacket] = []
        leads: list[SourceLead] = []
        statements: list[DiscoveryStatement] = []
        for result in results:
            self.session.tasks.append(result.task)
            if not result.artifact.payload:
                self.session.artifacts.append(result.artifact)
                continue
            packet = EvidencePacket.model_validate(result.artifact.payload)
            # Search grounding hands back a redirector, never a document. It is
            # followed here, before the packet is recorded, so the artifact and
            # every lead drawn from it name the paper rather than the link that
            # points at it.
            packet = await self._resolved_locators(packet)
            angle = angle_by_result.get(id(result))
            facet = angle.key if angle and angle.key in EVIDENCE_FACETS else ""
            relations_by_source: dict[str, list[str]] = {}
            for claim in packet.claims:
                if claim.source_id:
                    relations_by_source.setdefault(claim.source_id, []).append(
                        claim.relation
                    )
            if facet:
                for source in packet.sources:
                    source.facet = facet
            # Every search contributes its statements, whether or not it was
            # aimed at one of the seven facets. Only the facet tag depends on
            # that. A revision's own target -- what the researcher wrote in the
            # box -- carries no facet, and skipping it here meant the common
            # revision, the one where the audit named no gaps and the request is
            # the only search, contributed no statements at all. Coverage scores
            # research directions by counting statements, so a live revision
            # scored every direction zero and the panel reported coverage
            # falling from 88% to 70% for adding sources to the corpus.
            statements.extend(
                DiscoveryStatement(
                    text=claim.claim,
                    facet=facet,
                    source_urls=[
                        source.url
                        for source in packet.sources
                        if source.id == claim.source_id
                    ],
                    originating_pass=1,
                    relation=claim.relation,
                )
                for claim in packet.claims
            )
            result.artifact.payload = packet.model_dump(mode="json")
            self.session.artifacts.append(result.artifact)
            packets.append(packet)
            leads = merge_leads(
                leads,
                [
                    SourceLead(
                        canonical_url=source.url,
                        title=source.title,
                        source_type=source.source_type,
                        provider="google_search",
                        # What the locator states, so a paper the corpus already
                        # holds under its DOI is recognised as that paper rather
                        # than added again under its address.
                        identifiers={
                            **stated_identifiers(source.url),
                            **source.identifiers,
                        },
                        originating_passes=[1],
                        originating_statement_ids=list(source.supports_claim_ids),
                        facets=[facet] if facet else [],
                        claim_relations=list(
                            dict.fromkeys(relations_by_source.get(source.id, []))
                        ),
                        raw_artifact_reference=result.artifact.id,
                    )
                    for source in packet.sources
                ],
            )
        corpus = self._merged_discovery_corpus(packets, results, feedback=feedback)
        coverage = audit_coverage(
            DiscoveryNarrative(
                question=self.session.question,
                research_directions=list(plan.success_criteria),
                statements=statements,
            ),
            leads,
            # Which facets were searched is known from the angles that were
            # dispatched, so an angle that came back with nothing is scored as
            # a searched, empty facet rather than as one never asked about.
            searched_facets={
                angle.key for angle in angles if angle.key in EVIDENCE_FACETS
            },
        )
        manifest = DiscoveryManifest(
            discovery_angles=[angle.key for angle in angles],
            question=self.session.question,
            runs=[
                DeepResearchRun(
                    pass_number=1,
                    status="failed",
                    error=(
                        transport_error
                        or "Deep Research is not configured for this session."
                    ),
                    completed_at=utc_now(),
                )
            ],
            source_leads=leads,
            coverage_history=[coverage] if leads else [],
            convergence_reason=(
                "search_grounded_fallback" if leads else "deep_research_unavailable"
            ),
            verification_handoff_source_ids=[lead.id for lead in leads],
            stored_interaction_notice=False,
        )
        discovery = Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            artifact_type="specialist_output",
            content=self._evidence_summary(manifest),
            feedback=feedback,
            producer_model="google_search_grounding" if leads else "unavailable",
            schema_name="DiscoveryManifest",
            payload=manifest.model_dump(mode="json"),
            input_artifact_ids=[result.artifact.id for result in results]
            + ([corpus.id] if corpus is not None else []),
        )
        self.session.artifacts.append(discovery)
        return manifest, discovery

    async def _resolved_locators(self, packet: EvidencePacket) -> EvidencePacket:
        """Follow the packet's redirectors, and carry on if the network will not.

        A discovery pass that found real literature must not be lost because the
        links to it could not be dereferenced. What cannot be followed stays a
        redirector, and the verification stage records it as inaccessible.
        """
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                return await resolve_packet_locators(packet, client=client)
        except Exception:
            return packet

    async def _resolved_lead_locators(
        self, manifest: DiscoveryManifest
    ) -> DiscoveryManifest:
        """The same for a Deep Research manifest, degrading the same way."""
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                return await resolve_manifest_locators(manifest, client=client)
        except Exception:
            return manifest

    def _record_discovered_corpus(
        self, manifest: DiscoveryManifest, *, feedback: str
    ) -> Artifact | None:
        """Write down what the search found, as claims against the documents.

        The grounded path's specialist returns a packet and this is already
        there. Deep Research returns a manifest, and until now nothing turned
        one into the other: the verifier was handed titles and URLs, invented
        the handful of claims it could from them, and a run holding ninety-two
        verified sources reached the panel with ten findings to cite. That is
        the whole of "the knowledge base is thin" -- not what the search found,
        but what survived the handover.

        Merged with the corpus already standing rather than replacing it, so a
        revision that gap-searched keeps the findings that search returned.
        """
        corpus = discovered_corpus(self.session.question, manifest)
        # Nothing to add. A grounded manifest keeps no narratives -- its
        # specialist wrote the corpus directly and it is already standing -- so
        # this is the path that writes one and the path that leaves it alone.
        if not corpus.claims:
            return None
        prior = self._discovered_corpus()
        if prior is not None:
            corpus = merge_evidence_packets(
                self.session.question,
                [EvidencePacket.model_validate(prior.payload), corpus],
            )
            prior.status = ArtifactStatus.SUPERSEDED
        artifact = Artifact(
            stage="evidence",
            agent="evidence_discovery",
            artifact_type="specialist_output",
            content=(
                f"### Discovered corpus\n\n{len(corpus.sources)} distinct sources "
                f"and {len(corpus.claims)} findings, read off the search reports. "
                "Nothing here is verified."
            ),
            feedback=feedback,
            producer_model=getattr(self.provider, "model_id", "unknown"),
            schema_name="EvidencePacket",
            payload=corpus.model_dump(mode="json"),
        )
        self.session.artifacts.append(artifact)
        return artifact

    def _merged_discovery_corpus(
        self,
        packets: list[EvidencePacket],
        results: list,
        *,
        feedback: str,
    ) -> Artifact | None:
        """Fold the per-angle packets into the one artifact verification reads.

        Without this the angles are ten drafts with colliding identifiers, and
        the verifier is handed the manifest instead -- a list of titles and URLs
        with every claim discovery found stripped out of it. That is what the
        stage used to do with its single packet, and it is why the verifier
        re-invented a handful of claims rather than checking the ones the search
        actually returned.

        The angle artifacts are superseded rather than deleted: they are the
        record of which search found what, and the appendix cites them.
        """
        if not packets:
            return None
        merged = merge_evidence_packets(self.session.question, packets)
        corpus = Artifact(
            stage="evidence",
            agent="evidence_discovery",
            artifact_type="specialist_output",
            content=(
                f"### Discovered corpus\n\n{len(merged.sources)} distinct sources "
                f"and {len(merged.claims)} claims, merged from "
                f"{len(packets)} search angles. Nothing here is verified."
            ),
            feedback=feedback,
            producer_model=getattr(self.provider, "model_id", "unknown"),
            schema_name="EvidencePacket",
            payload=merged.model_dump(mode="json"),
            input_artifact_ids=[result.artifact.id for result in results],
        )
        for result in results:
            result.artifact.status = ArtifactStatus.SUPERSEDED
        self.session.artifacts.append(corpus)
        return corpus

    @staticmethod
    def _evidence_summary(manifest: DiscoveryManifest) -> str:
        latest = manifest.coverage_history[-1] if manifest.coverage_history else None
        coverage = f"{latest.weighted_score:.0%}" if latest else "not available"
        gaps = latest.gaps if latest else []
        # A pass counts as completed once its report has been read into the
        # manifest, not when the provider says it finished. The two came apart on
        # a live run that lost six of seven reports to a restart: every pass was
        # marked completed and the card said "8 completed of 8 attempted" over a
        # corpus built from two of them.
        lost = unread_passes(manifest)
        completed = len(
            [
                run
                for run in manifest.runs
                if run.status == "completed" and run.pass_number not in lost
            ]
        )
        unread = len(lost)
        providers = sorted({lead.provider for lead in manifest.source_leads})
        # Read off the leads, so a wave that returned none has none to name. Printed
        # as a bare "none" directly under a line reporting seven attempted passes, it
        # says the deployment has no discovery provider configured, which is a broken
        # install rather than a search that came back empty.
        named = ", ".join(providers) or (
            "none named -- no pass returned a lead"
            if manifest.runs
            else "none -- no pass was attempted"
        )
        lines = [
            "### Evidence Discovery",
            "",
            f"- Deep Research passes: {completed} completed of "
            f"{len(manifest.runs)} attempted (limit {MAX_DEEP_RESEARCH_PASSES})"
            + (
                ""
                if not unread
                else f"; {unread} finished and could not be read back, so nothing "
                "from them is in this corpus"
            ),
            f"- Discovery provider: {named}",
            f"- Coverage: {coverage}",
            f"- Source leads: {len(manifest.source_leads)}",
            f"- Estimated cost: ${manifest.estimated_cost_usd:.2f}",
            f"- Stop reason: {manifest.convergence_reason or 'in progress'}",
            "- Status: discovered, not yet verified",
        ]
        if manifest.leads_beyond_retention_ceiling:
            # Same reason as the batch ceiling below. Every count on the panel
            # and every facet on the evidence floor is computed from what
            # survived this cut, so the cut has to appear next to them.
            lines.append(
                f"- Not retained: {manifest.leads_beyond_retention_ceiling} "
                "further leads, beyond what one manifest holds; the retained "
                "leads are spread across the facets that were searched"
            )
        if manifest.leads_sent_to_verification:
            lines.append(
                f"- Sent to verification: {manifest.leads_sent_to_verification} "
                "of the leads that name a document"
            )
        if manifest.leads_beyond_verification_ceiling:
            # A truncation nobody states reads as coverage of everything.
            lines.append(
                f"- Not verified: {manifest.leads_beyond_verification_ceiling} "
                "further leads, left unchecked because the batch ceiling was "
                "reached"
            )
        gap_searches = len(
            [
                request
                for request in manifest.enrichment_requests
                if request.status == "completed"
            ]
        )
        failed_searches = len(
            [
                request
                for request in manifest.enrichment_requests
                if request.status == "failed"
            ]
        )
        if gap_searches:
            lines.append(
                f"- Gap-directed searches: {gap_searches} run against named gaps "
                "in this corpus, without a further Deep Research pass"
            )
        if failed_searches:
            # Counted apart from the completed ones rather than left out of both
            # lines. A revision that dispatched six searches and got two back
            # otherwise reports two, and the four that returned nothing are
            # indistinguishable from searches nobody ran.
            lines.append(
                f"- Gap-directed searches that returned nothing: {failed_searches}"
            )
        if manifest.gap_searches_deferred:
            # The gaps are listed below either way, so without this line one
            # that was searched and came back empty reads exactly like one this
            # revision never got to.
            lines.append(
                f"- Gaps not searched: {manifest.gap_searches_deferred}, beyond "
                "what one revision runs; they are still listed as gaps below"
            )
        if manifest.stored_interaction_notice:
            lines.append(
                "- Stored interaction notice: Deep Research uses stored Gemini "
                "interactions."
            )
        if gaps:
            lines.extend(["", "#### Unresolved gaps"])
            lines.extend(f"- {gap.description}" for gap in gaps)
        if any(run.error for run in manifest.runs):
            lines.extend(["", "#### Run errors"])
            lines.extend(
                f"- Pass {run.pass_number}: {run.error}"
                for run in manifest.runs
                if run.error
            )
        return "\n".join(lines)

    def preview(self, feedback: str = "") -> Artifact:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.apreview(feedback))
        raise RuntimeError(
            "Use 'await workflow.apreview()' inside an async event loop."
        )

    def accept(
        self,
        artifact: Artifact,
        *,
        actor: str = "researcher",
        automatic: bool | None = None,
    ) -> None:
        # ``evidence_required`` is a finding about the corpus, not a lock on the
        # session. The gate below re-measures the floor and refuses again where
        # the corpus is still short, so the status adds no safety by also
        # barring the door -- and barring it left a run whose floor was met but
        # miscounted with nowhere to go: the only ways out of that status waive
        # the floor or re-run discovery, and neither is what a researcher
        # looking at a corpus that does clear it is asking for.
        retryable = (
            self.session.status == "evidence_required" and artifact.stage == "evidence"
        )
        if self.done or (self.session.status != "active" and not retryable):
            raise ValueError("The session cannot accept another stage.")
        pending = self.pending_draft
        if pending is None or artifact.id != pending.id:
            raise ValueError("Only the current, latest stage draft can be accepted.")
        if artifact.stage != self.stage or artifact.version != pending.version:
            raise ValueError("Stale or wrong-stage approval rejected.")
        if artifact.stage == "scope" and unresolved_blockers(self.session):
            self.session.status = "input_required"
            event = self._event(
                "input_required",
                "supervisor",
                payload={
                    "requirement_ids": [
                        item.id for item in unresolved_blockers(self.session)
                    ]
                },
            )
            self._persist(event)
            raise ValueError(
                "Required scientific input is missing. Provide it or explicitly "
                "select the permitted literature-only fallback."
            )
        if (
            artifact.stage == "evidence"
            and not self.session.exploratory_evidence_accepted
        ):
            inputs = [
                item
                for item in self.session.artifacts
                if item.id in artifact.input_artifact_ids
            ]
            manifest_artifact = next(
                (item for item in inputs if item.schema_name == "DiscoveryManifest"),
                None,
            )
            # The newest packet, not the first one listed. Verification runs in
            # batches of twelve leads and appends a packet per batch before
            # appending the corpus that merges them, so a forward scan reads
            # batch one -- twelve leads of sixty-four -- and refuses a gate that
            # the panel beside it reports as met. A live run stopped on "0 of 8
            # weighted verified sources" while its own evidence page read
            # "Twenty-six usable sources across seven facets". Everything else
            # downstream reads the newest packet; the merge supersedes the
            # batches, and preferring a standing one says which is newest even
            # where the inputs are not in the order they were appended.
            packets = [
                item
                for item in reversed(inputs)
                if item.schema_name == "EvidencePacket"
            ]
            packet_artifact = next(
                (item for item in packets if item.status != ArtifactStatus.SUPERSEDED),
                packets[0] if packets else None,
            )
            manifest = (
                DiscoveryManifest.model_validate(manifest_artifact.payload)
                if manifest_artifact
                else None
            )
            manifest_ok = bool(manifest and manifest.source_leads)
            packet = (
                EvidencePacket.model_validate(packet_artifact.payload)
                if packet_artifact
                else None
            )
            # The test here used to be that every source in the packet was
            # verified. That is unclearable by construction -- one publisher
            # being down on the day fails a corpus of ninety -- and a gate that
            # can only ever refuse teaches people to click past it. The floor
            # measures the three things that decide whether a hypothesis rests
            # on a literature: how much of it was checked, how many kinds of
            # evidence were found, and whether anything was found that
            # disagrees.
            floor = (
                evaluate_evidence_floor(packet, manifest)
                if packet
                else EvidenceFloor(shortfalls=["No verification packet was produced."])
            )
            if manifest_ok and floor.met:
                # Entered from ``evidence_required``, this is the refusal being
                # withdrawn, and the status has to go with it or the run is
                # still parked on a finding that no longer holds.
                self.session.status = "active"
            elif self.session.rehearsal:
                # The other gate a rehearsal has no person for, answered the
                # same way: in writing, under a name nobody could have typed,
                # and with what the gate reported carried into the record rather
                # than dropped. A live rehearsal launched to exercise the
                # pipeline end to end stopped here at evidence_required, which
                # is the floor doing its job over a corpus nobody was going to
                # act on.
                self.session.exploratory_evidence_accepted = True
                self.session.status = "active"
                self._persist(
                    self._event(
                        # The event the waived-gate advisory reads, so a
                        # rehearsal's waiver reaches the report by the path
                        # every other waiver reaches it by. What it says there
                        # is not the same, because the actor is not.
                        "limited_exploratory_evidence_accepted",
                        REHEARSAL_ADJUDICATOR,
                        payload={
                            "manifest_ok": manifest_ok,
                            "evidence_floor": floor.model_dump(mode="json"),
                            "warning": (
                                "All downstream outputs remain hypotheses and must "
                                "not be presented as evidence-backed findings."
                            ),
                        },
                    )
                )
            else:
                self.session.status = "evidence_required"
                self._persist(
                    self._event(
                        "evidence_verification_required",
                        "supervisor",
                        payload={
                            "manifest_ok": manifest_ok,
                            "packet_ok": floor.met,
                            "evidence_floor": floor.model_dump(mode="json"),
                        },
                    )
                )
                raise ValueError(
                    "The evidence base does not meet the floor for generating "
                    "hypotheses. "
                    + (
                        " ".join(floor.shortfalls)
                        if floor.shortfalls
                        else "No source leads were discovered."
                    )
                    + " Retry discovery, or explicitly select the limited "
                    "exploratory workflow to proceed on this evidence."
                )
        if (
            self.approval_profile == ApprovalProfile.ARTIFACT
            and self.pending_artifact_reviews
        ):
            raise ValueError(
                "Every specialist artifact must be approved before the stage bundle."
            )
        if artifact.stage == "reflect":
            # Only findings nobody has answered still block. An adjudicated one
            # has been withdrawn or explicitly accepted by a named person, and
            # re-raising it would make the gate impossible to clear.
            unanswered = [
                blocker
                for blocker in open_blockers(self.session)
                if blocker.artifact_id in artifact.input_artifact_ids
            ]
            if unanswered and self.session.rehearsal:
                # A rehearsal answers its own gate rather than parking here for
                # a person it does not have. Recorded, not skipped: each finding
                # gets an override in the session saying in writing that nobody
                # read it, so the dossier prints the flaw and the non-answer
                # together. See ``rehearsal_adjudications``.
                self.session.governance_adjudications.extend(
                    rehearsal_adjudications(unanswered)
                )
                self._persist(
                    self._event(
                        "governance_waived_for_rehearsal",
                        REHEARSAL_ADJUDICATOR,
                        payload={
                            "review_ids": [blocker.review_id for blocker in unanswered],
                            "candidate_ids": [
                                blocker.candidate_id for blocker in unanswered
                            ],
                        },
                    )
                )
                unanswered = []
            if unanswered:
                self.session.status = "governance_blocked"
                event = self._event(
                    "governance_blocked",
                    "supervisor",
                    payload={
                        "review_ids": [blocker.review_id for blocker in unanswered],
                        "candidate_ids": [
                            blocker.candidate_id for blocker in unanswered
                        ],
                    },
                )
                self._persist(event)
                return

        if (
            artifact.stage in {"rank", "meta_review"}
            and self.session.workflow_version >= 2
        ):
            all_reviews = [
                review
                for item in self.session.artifacts
                if item.stage == "reflect" and item.schema_name == "ReviewSet"
                for review in ReviewSet.model_validate(item.payload).reviews
            ]
            if all_reviews and all(
                review.recommendation == "insufficient_evidence"
                for review in all_reviews
            ):
                raise ValueError(
                    "A review set in which all candidates have insufficient evidence "
                    "cannot produce a scientific recommendation."
                )

        is_automatic = (
            self.approval_profile == ApprovalProfile.AUTO
            if automatic is None
            else automatic
        )
        automatic_allowed = (
            self.approval_profile == ApprovalProfile.AUTO
            or (
                self.approval_profile == ApprovalProfile.MILESTONE
                and not self.requires_human_approval
            )
            or self.approval_profile == ApprovalProfile.ARTIFACT
        )
        if is_automatic and not automatic_allowed:
            raise ValueError("Automatic decisions are disabled in human mode.")
        decision_actor = (
            "auto_approval_policy"
            if self.approval_profile == ApprovalProfile.AUTO and is_automatic
            else "milestone_auto_policy"
            if is_automatic
            else actor
        )

        for item in self.session.artifacts:
            if item.id in artifact.input_artifact_ids or item.id == artifact.id:
                item.status = ArtifactStatus.ACCEPTED
            elif (
                item.stage == artifact.stage
                and item.status == ArtifactStatus.DRAFT
                and item.id != artifact.id
            ):
                item.status = ArtifactStatus.SUPERSEDED
        decision = HumanDecision(
            action=DecisionAction.ACCEPT,
            artifact_id=artifact.id,
            artifact_version=artifact.version,
            stage=artifact.stage,
            actor=decision_actor,
            automatic=is_automatic,
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        self.session.current_stage += 1
        if self.done:
            self.session.status = "ready_for_report"
        event = self._event(
            "stage_accepted",
            decision_actor,
            payload={
                "artifact_id": artifact.id,
                "automatic": is_automatic,
                "decision_id": decision.id,
            },
            stage=artifact.stage,
        )
        self._persist(event)

    def approve_artifact(
        self, artifact: Artifact, *, actor: str = "researcher"
    ) -> None:
        """Approve one specialist result in the artifact-level profile."""
        if self.approval_profile != ApprovalProfile.ARTIFACT:
            raise ValueError("Artifact decisions require approval_profile='artifact'.")
        if artifact not in self.pending_artifact_reviews:
            raise ValueError(
                "Only a current, undecided specialist artifact can be approved."
            )
        artifact.status = ArtifactStatus.ACCEPTED
        decision = HumanDecision(
            action=DecisionAction.ACCEPT,
            artifact_id=artifact.id,
            artifact_version=artifact.version,
            stage=artifact.stage,
            actor=actor,
            automatic=False,
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        event = self._event(
            "specialist_artifact_accepted",
            actor,
            payload={"artifact_id": artifact.id, "decision_id": decision.id},
        )
        self._persist(event)

    def accept_literature_only(self, *, actor: str = "researcher") -> None:
        """Resolve eligible missing inputs without pretending analysis occurred."""
        eligible = [
            requirement
            for requirement in unresolved_blockers(self.session)
            if requirement.permitted_fallback == "literature_only"
        ]
        if not eligible:
            raise ValueError("No unresolved input supports a literature-only fallback.")
        for requirement in eligible:
            requirement.status = "fallback_accepted"
        self.session.literature_only = True
        self.session.status = "active"
        event = self._event(
            "literature_only_fallback_accepted",
            actor,
            payload={"requirement_ids": [item.id for item in eligible]},
        )
        self._persist(event)

    def accept_exploratory_evidence(self, *, actor: str = "researcher") -> None:
        """Explicitly waive verified-evidence generation, never automatically."""
        if self.stage != "evidence":
            raise ValueError("Exploratory fallback is available only at Evidence.")
        self.session.exploratory_evidence_accepted = True
        self.session.status = "active"
        self._persist(
            self._event(
                "limited_exploratory_evidence_accepted",
                actor,
                payload={
                    "warning": (
                        "All downstream outputs remain hypotheses and must not be "
                        "presented as evidence-backed findings."
                    )
                },
            )
        )

    def adjudicate_governance(
        self,
        review_id: str,
        resolution: str,
        *,
        adjudicator: str,
        justification: str,
    ) -> GovernanceAdjudication:
        """Answer one fatal governance finding so the session can move again.

        Deliberately one finding at a time. A single command that cleared every
        block at once would let one sentence of justification stand for several
        unrelated hazards, and the point of the record is that each flaw was
        read by somebody who then said what they were doing about it.
        """
        if self.session.status != "governance_blocked":
            raise ValueError(
                "Governance adjudication applies only to a blocked session."
            )
        if resolution not in {"withdraw", "override"}:
            raise ValueError(
                f"Unknown governance resolution: {resolution!r}. "
                "Use 'withdraw' to drop the hypothesis or 'override' to accept "
                "the flaw on the record."
            )
        blocker = next(
            (
                item
                for item in open_blockers(self.session)
                if item.review_id == review_id
            ),
            None,
        )
        if blocker is None:
            if review_id in adjudicated_review_ids(self.session):
                raise ValueError(f"Review {review_id} has already been adjudicated.")
            open_ids = ", ".join(item.review_id for item in open_blockers(self.session))
            raise ValueError(
                f"Review {review_id} is not an open governance blocker. "
                f"Open blockers: {open_ids or 'none'}."
            )

        replacement = None
        if resolution == "withdraw":
            # Raised out of withdraw_candidate before anything is recorded, so a
            # refused withdrawal leaves the session exactly as it was.
            replacement = withdraw_candidate(self.session, blocker.candidate_id)
        adjudication = record_adjudication(
            self.session,
            blocker,
            resolution=resolution,
            adjudicator=adjudicator,
            justification=justification,
        )

        remaining = open_blockers(self.session)
        if not remaining:
            self.session.status = "active"
        event = self._event(
            "governance_adjudicated",
            adjudicator,
            payload={
                "review_id": review_id,
                "candidate_id": blocker.candidate_id,
                "resolution": resolution,
                "justification": justification,
                "revised_population_artifact_id": (
                    replacement.id if replacement else None
                ),
                "remaining_blocker_ids": [item.review_id for item in remaining],
            },
        )
        self._persist(event)
        return adjudication

    def retry_evidence(self, *, actor: str = "researcher") -> None:
        if self.stage != "evidence":
            raise ValueError("Evidence retry is available only at Evidence.")
        pending = self.pending_draft
        if pending is not None:
            pending.status = ArtifactStatus.SUPERSEDED
        self.session.status = "active"
        self._persist(
            self._event(
                "evidence_retry_requested",
                actor,
                payload={"superseded_artifact_id": pending.id if pending else None},
            )
        )

    def provide_input(
        self, input_type: str, reference: str, *, actor: str = "researcher"
    ) -> None:
        requirement = next(
            (
                item
                for item in self.session.input_requirements
                if item.input_type == input_type and not item.resolved
            ),
            None,
        )
        if requirement is None:
            raise ValueError(f"No unresolved input requirement: {input_type}")
        if not reference.strip():
            raise ValueError("An input reference is required.")
        requirement.status = "provided"
        requirement.provided_reference = reference.strip()
        if not unresolved_blockers(self.session):
            self.session.status = "active"
        event = self._event(
            "research_input_provided",
            actor,
            payload={"requirement_id": requirement.id, "reference": reference.strip()},
        )
        self._persist(event)

    def request_revision(self, feedback: str, *, actor: str = "researcher") -> Artifact:
        """Record revision intent without synchronously generating its replacement."""
        if not feedback.strip():
            raise ValueError("Revision feedback is required.")
        pending = self.pending_draft
        if pending is None:
            raise ValueError("There is no current stage draft to revise.")
        pending.status = ArtifactStatus.SUPERSEDED
        decision = HumanDecision(
            action=DecisionAction.REVISE,
            artifact_id=pending.id,
            artifact_version=pending.version,
            stage=pending.stage,
            actor=actor,
            feedback=feedback,
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        event = self._event(
            "stage_revision_requested",
            actor,
            payload={"artifact_id": pending.id, "feedback": feedback},
        )
        self._persist(event)
        return pending

    def revise(self, feedback: str, *, actor: str = "researcher") -> Artifact:
        self.request_revision(feedback, actor=actor)
        return self.preview(feedback)

    def edit_draft(self, content: str, *, actor: str = "researcher") -> Artifact:
        """Create an auditable human-edited version of the current stage bundle."""
        if not content.strip():
            raise ValueError("Edited draft content is required.")
        pending = self.pending_draft
        if pending is None:
            raise ValueError("There is no current stage draft to edit.")
        pending.status = ArtifactStatus.SUPERSEDED
        edited = Artifact(
            stage=pending.stage,
            agent="supervisor",
            content=content.strip(),
            feedback="Direct researcher edit",
            version=pending.version + 1,
            parent_id=pending.id,
            producer_model="human-edited",
            input_artifact_ids=list(pending.input_artifact_ids),
        )
        self.session.artifacts.append(edited)
        decision = HumanDecision(
            action=DecisionAction.REVISE,
            artifact_id=pending.id,
            artifact_version=pending.version,
            stage=pending.stage,
            actor=actor,
            feedback="Direct researcher edit",
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        event = self._event(
            "stage_directly_edited",
            actor,
            payload={
                "artifact_id": edited.id,
                "parent_id": pending.id,
                "decision_id": decision.id,
            },
        )
        self._persist(event)
        return edited

    STOPPABLE_STATUSES = frozenset(
        {"active", "input_required", "evidence_required", "governance_blocked"}
    )
    """States a human can still walk away from, as opposed to end states."""

    def refine_section(
        self,
        candidate_id: str,
        section: str,
        feedback: str,
        *,
        actor: str = "researcher",
    ) -> Artifact:
        """Refine a single field of a candidate card without regenerating the population."""
        pending = self.pending_draft
        if pending is None or pending.stage != "generate":
            raise ValueError(
                "Targeted section refinement is available only during candidate generation."
            )
        pop_artifact = None
        for art in reversed(self.session.artifacts):
            if (
                art.stage == "generate"
                and art.schema_name == "CandidatePopulation"
                and art.payload
            ):
                pop_artifact = art
                break
        if not pop_artifact:
            raise ValueError("No CandidatePopulation artifact found to refine.")
        pop = CandidatePopulation.model_validate(pop_artifact.payload)
        target_cand = None
        for cand in pop.candidates:
            if cand.id == candidate_id:
                target_cand = cand
                break
        if not target_cand:
            raise ValueError(
                f"Candidate {candidate_id} not found in current population."
            )
        if not hasattr(target_cand, section):
            raise ValueError(f"Candidate has no section '{section}'.")
        current_val = getattr(target_cand, section)
        if isinstance(current_val, str):
            setattr(
                target_cand,
                section,
                f"{current_val}\n\nResearcher refinement ({feedback})",
            )
        elif isinstance(current_val, list):
            current_val.append(f"Researcher refinement: {feedback}")
        pop_artifact.revise(pop_artifact.content, pop.model_dump(mode="json"))
        decision = HumanDecision(
            action=DecisionAction.REFINE_SECTION,
            artifact_id=pop_artifact.id,
            artifact_version=pop_artifact.version,
            stage=pop_artifact.stage,
            actor=actor,
            feedback=f"Refined section '{section}' of candidate '{candidate_id}': {feedback}",
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        event = self._event(
            "candidate_section_refined",
            actor,
            payload={
                "candidate_id": candidate_id,
                "section": section,
                "feedback": feedback,
                "artifact_id": pop_artifact.id,
                "decision_id": decision.id,
            },
        )
        self._persist(event)
        return pop_artifact

    def request_evidence_delta(
        self,
        requesting_stage: str,
        requesting_agent: str,
        claim_to_verify: str,
        *,
        priority: int = 1,
        budget_usd: float = 1.0,
    ) -> EvidenceRequest:
        """Submit an asynchronous evidence request during review or evolution."""
        searches_used = sum(
            1
            for event in self.session.events
            if event.event_type == "evidence_delta_requested"
        )
        for item in self.session.artifacts:
            if item.schema_name == "DiscoveryManifest" and item.payload:
                dm = DiscoveryManifest.model_validate(item.payload)
                searches_used += len(dm.runs)
            elif item.schema_name == "KnowledgeBaseManifest" and item.payload:
                kb = KnowledgeBaseManifest.model_validate(item.payload)
                searches_used += len(kb.evidence_requests)

        if searches_used >= self.session.budget.max_searches:
            raise ValueError(
                "ResearchBudget max_searches exceeded for evidence delta requests."
            )

        ev_req = EvidenceRequest(
            requesting_stage=requesting_stage,
            requesting_agent=requesting_agent,
            claim_to_verify=claim_to_verify,
            priority=priority,
            budget_usd=budget_usd,
            status="submitted",
        )
        kb_artifact = next(
            (
                item
                for item in reversed(self.session.artifacts)
                if item.schema_name == "KnowledgeBaseManifest"
            ),
            None,
        )
        if kb_artifact:
            kb = KnowledgeBaseManifest.model_validate(kb_artifact.payload)
            kb.evidence_requests.append(ev_req)
            kb_artifact.revise(
                f"Knowledge Base Manifest v{kb.version} with "
                f"{len(kb.evidence_requests)} evidence requests.",
                kb.model_dump(mode="json"),
            )
        else:
            kb = KnowledgeBaseManifest(version=1, evidence_requests=[ev_req])
            kb_artifact = Artifact(
                stage="evidence",
                agent="knowledge_curator",
                content="Knowledge Base Manifest v1 with 1 evidence request.",
                schema_name="KnowledgeBaseManifest",
                payload=kb.model_dump(mode="json"),
            )
            self.session.artifacts.append(kb_artifact)
        self._persist(
            self._event(
                "evidence_delta_requested",
                requesting_agent,
                payload={"evidence_request_id": ev_req.id, "claim": claim_to_verify},
            )
        )
        return ev_req

    def stop(self, *, actor: str = "researcher") -> None:
        """Record that a human ended the session, including at a gate.

        This returned silently unless the session was active, so the two states
        an operator is most likely to walk away from -- an unmet evidence gate,
        an open fatal safety finding -- recorded nothing at all: no decision, no
        audit event, and a saved status still reading as though the run were
        waiting for an answer. The terminal said "session stopped" and the file
        disagreed. What was left unanswered is now part of the record.
        """
        if self.session.status not in self.STOPPABLE_STATUSES:
            return
        halted_at = self.session.status
        unanswered = [blocker.review_id for blocker in open_blockers(self.session)]
        decision = HumanDecision(
            action=DecisionAction.STOP,
            stage=self.stage,
            actor=actor,
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        self.session.status = "stopped_by_researcher"
        event = self._event(
            "session_stopped",
            actor,
            payload={
                "decision_id": decision.id,
                "halted_at": halted_at,
                "unanswered_governance_findings": unanswered,
            },
        )
        self._persist(event)

    def run_auto(self) -> None:
        if self.approval_profile != ApprovalProfile.AUTO:
            raise ValueError("run_auto requires approval_profile='auto'.")
        while not self.done and self.session.status == "active":
            try:
                self.accept(self.preview(), automatic=True)
            except ValueError:
                if self.session.status == "evidence_required":
                    break
                raise

    def advance_to_human_gate(self) -> None:
        """Advance bounded internal work for the milestone interaction profile."""
        if self.approval_profile != ApprovalProfile.MILESTONE:
            return
        while (
            not self.done
            and self.session.status == "active"
            and not self.requires_human_approval
        ):
            try:
                self.accept(self.preview(), automatic=True)
            except ValueError:
                if self.session.status == "evidence_required":
                    break
                raise

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.session.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        provider: Provider | None = None,
        *,
        ledger: ResearchLedger | None = None,
        approval_mode: ApprovalMode | str | None = None,
        approval_profile: ApprovalProfile | str | None = None,
    ) -> CoScientistWorkflow:
        session = Session.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if approval_profile is not None:
            session.approval_profile = ApprovalProfile(approval_profile)
            session.approval_mode = (
                ApprovalMode.AUTO
                if session.approval_profile == ApprovalProfile.AUTO
                else ApprovalMode.HUMAN
            )
        elif approval_mode is not None:
            session.approval_mode = ApprovalMode(approval_mode)
            session.approval_profile = (
                ApprovalProfile.AUTO
                if session.approval_mode == ApprovalMode.AUTO
                else ApprovalProfile.STAGE
            )
        return cls(
            session.question,
            provider,
            session,
            ledger=ledger,
        )

    @classmethod
    def load_from_ledger(
        cls,
        session_id: str,
        ledger: ResearchLedger,
        provider: Provider | None = None,
    ) -> CoScientistWorkflow:
        session = ledger.load(session_id)
        return cls(session.question, provider, session, ledger=ledger)

    def render_report(self) -> str:
        return compile_dossier(self.session)

    def report_filename(self) -> str:
        slug = (
            re.sub(r"[^a-z0-9]+", "-", self.session.question.lower()).strip("-")[:48]
            or "research"
        )
        return f"coscientist-{slug}.md"
