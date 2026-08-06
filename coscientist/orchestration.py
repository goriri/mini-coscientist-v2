"""Durable, code-enforced Supervisor for the scientific workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
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
from .dossier import compile_dossier
from .evidence import (
    EvidenceArtifactStore,
    EvidenceStillRunning,
    GeminiDeepResearchTransport,
    GeminiEvidenceNormalizer,
    IterativeEvidenceDiscovery,
    RegistryMetadataEnricher,
    audit_coverage,
    discovery_angles,
    downgrade_unlocatable_sources,
    evaluate_evidence_floor,
    merge_evidence_packets,
    merge_leads,
    resolve_manifest_locators,
    resolve_packet_locators,
    sweep_verification,
)
from .governance import (
    adjudicated_review_ids,
    open_blockers,
    record_adjudication,
    withdraw_candidate,
)
from .ledger import ResearchLedger
from .methods import classify_research_mode, method_requirements
from .model_catalog import DEFAULT_LANGUAGE, DEFAULT_MODEL
from .models import (
    EVIDENCE_FACETS,
    STAGES,
    ApprovalMode,
    ApprovalProfile,
    Artifact,
    ArtifactStatus,
    AuditEvent,
    DecisionAction,
    DeepResearchRun,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EvidenceFloor,
    EvidencePacket,
    GovernanceAdjudication,
    HumanDecision,
    ResearchPlan,
    Session,
    SourceLead,
    utc_now,
)
from .parity import detect_input_requirements, unresolved_blockers

WORKFLOW_STAGES_V1 = tuple(
    stage for stage in STAGES if stage not in {"evidence", "report"}
)
WORKFLOW_STAGES = tuple(stage for stage in STAGES if stage != "report")
MILESTONE_STAGES = frozenset({"scope", "rank", "evolve", "meta_review"})


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
        ledger: ResearchLedger | None = None,
        evidence_discovery: IterativeEvidenceDiscovery | None = None,
    ):
        self.provider = provider or DeterministicProvider()
        self.ledger = ledger
        self.evidence_discovery = evidence_discovery
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
                workflow_version=workflow_version,
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
        if not self.session.question:
            raise ValueError("A research question is required.")
        method_requirements(self.session.research_mode)
        # The session is the authority on which model this run uses. A caller
        # resuming one has to construct a provider before it can read the
        # session that answers the question, so the answer is applied here
        # rather than left to every call site to remember.
        bind_provider_model(self.provider, self.session.model)
        self.task_bus = LocalA2ATaskBus(SPECIALISTS, self.provider)
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
            return self.stage in MILESTONE_STAGES
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
            },
        )
        self._persist(event)
        return draft

    async def _preview_evidence(self, feedback: str = "") -> Artifact:
        """Run discovery first, then hand discovered leads to verification."""
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
            discovery = next(
                (
                    item
                    for item in reversed(self.session.artifacts)
                    if item.stage == "evidence"
                    and item.agent == "deep_research_discovery"
                    and item.schema_name == "DiscoveryManifest"
                    and item.status == ArtifactStatus.DRAFT
                ),
                None,
            )
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
        if manifest.enrichment_requests:
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
                        f"- {request.query}"
                        for request in manifest.enrichment_requests[:6]
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
        results = await self.task_bus.dispatch_stage(
            temporary,
            verifier_definition,
            feedback=feedback,
            revision=revision,
        )
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

        if verified_packets:
            manifest = self._manifest_with_verification(manifest, verified_packets)
            discovery.revise(
                self._evidence_summary(manifest), manifest.model_dump(mode="json")
            )

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
                },
            )
        )
        return draft

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
        existing = next(
            (
                item
                for item in reversed(self.session.artifacts)
                if item.stage == "evidence"
                and item.agent == "deep_research_discovery"
                and item.schema_name == "DiscoveryManifest"
                and item.status == ArtifactStatus.DRAFT
                and item.payload
            ),
            None,
        )
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
        completed = len([run for run in manifest.runs if run.status == "completed"])
        providers = sorted({lead.provider for lead in manifest.source_leads})
        lines = [
            "### Evidence Discovery",
            "",
            f"- Deep Research passes: {completed} completed of "
            f"{len(manifest.runs)} attempted (limit 3)",
            f"- Discovery provider: {', '.join(providers) or 'none'}",
            f"- Coverage: {coverage}",
            f"- Source leads: {len(manifest.source_leads)}",
            f"- Estimated cost: ${manifest.estimated_cost_usd:.2f}",
            f"- Stop reason: {manifest.convergence_reason or 'in progress'}",
            "- Status: discovered, not yet verified",
        ]
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
        if self.done or self.session.status != "active":
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
            packet_artifact = next(
                (item for item in inputs if item.schema_name == "EvidencePacket"),
                None,
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
            if not (manifest_ok and floor.met):
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
