"""Durable, code-enforced Supervisor for the scientific workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from .agents import (
    SPECIALISTS,
    SPECIALISTS_BY_STAGE,
    DeterministicProvider,
    Provider,
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
)
from .ledger import ResearchLedger
from .methods import classify_research_mode, method_requirements
from .models import (
    STAGES,
    ApprovalMode,
    ApprovalProfile,
    Artifact,
    ArtifactStatus,
    AuditEvent,
    DecisionAction,
    DeepResearchRun,
    DiscoveryManifest,
    EvidencePacket,
    HumanDecision,
    ResearchPlan,
    ReviewSet,
    Session,
)
from .parity import detect_input_requirements, unresolved_blockers

WORKFLOW_STAGES_V1 = tuple(
    stage for stage in STAGES if stage not in {"evidence", "report"}
)
WORKFLOW_STAGES = tuple(stage for stage in STAGES if stage != "report")
MILESTONE_STAGES = frozenset({"scope", "rank", "evolve", "meta_review"})


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
                workflow_version=workflow_version,
                input_requirements=detect_input_requirements(question),
            )
        else:
            self.session = session
        if not self.session.question:
            raise ValueError("A research question is required.")
        method_requirements(self.session.research_mode)
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
        if controller is None and os.environ.get("GEMINI_API_KEY"):
            repeat_enabled = (
                os.environ.get("EVIDENCE_REPEAT_PASSES", "false").lower() == "true"
            )
            pass_three_enabled = (
                os.environ.get("EVIDENCE_ENABLE_PASS_3", "false").lower() == "true"
            )
            controller = IterativeEvidenceDiscovery(
                GeminiDeepResearchTransport(),
                EvidenceArtifactStore(),
                max_passes=(
                    3
                    if repeat_enabled and pass_three_enabled
                    else 2
                    if repeat_enabled
                    else 1
                ),
                registry_enricher=RegistryMetadataEnricher(),
                polls_per_invocation=(
                    1
                    if os.environ.get("EVIDENCE_TASK_STEP_MODE", "false").lower()
                    == "true"
                    else None
                ),
            )
        if controller is None:
            manifest = DiscoveryManifest(
                question=self.session.question,
                runs=[
                    DeepResearchRun(
                        pass_number=1,
                        status="failed",
                        error="GEMINI_API_KEY is not configured for Deep Research.",
                    )
                ],
                convergence_reason="deep_research_unavailable",
            )
            discovery = Artifact(
                stage="evidence",
                agent="deep_research_discovery",
                artifact_type="specialist_output",
                content=self._evidence_summary(manifest),
                feedback=feedback,
                producer_model="unavailable",
                schema_name="DiscoveryManifest",
                payload=manifest.model_dump(mode="json"),
            )
            self.session.artifacts.append(discovery)
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
                discovery.payload = updated.model_dump(mode="json")
                discovery.content = self._evidence_summary(updated)
                self._persist()

            manifest = await asyncio.to_thread(
                controller.run,
                self.session,
                plan,
                manifest=existing_manifest,
                normalizer=(
                    GeminiEvidenceNormalizer()
                    if isinstance(controller.transport, GeminiDeepResearchTransport)
                    else None
                ),
                manifest_callback=persist_manifest,
            )
            if any(
                run.status in {"queued", "in_progress", "requires_action"}
                for run in manifest.runs
            ):
                raise EvidenceStillRunning(
                    "Deep Research interaction is still running."
                )

        summary = self._evidence_summary(manifest)
        discovery.content = summary
        discovery.payload = manifest.model_dump(mode="json")
        output_ids = [discovery.id]

        # Verification sees the immutable discovery manifest, but discovery itself
        # remains a draft until the complete Evidence stage is promoted.
        temporary = self.session.model_copy(deep=True)
        temp_discovery = next(
            item for item in temporary.artifacts if item.id == discovery.id
        )
        temp_discovery.status = ArtifactStatus.ACCEPTED
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
        for result in results:
            self.session.tasks.append(result.task)
            self.session.artifacts.append(result.artifact)
            output_ids.append(result.artifact.id)

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

    @staticmethod
    def _evidence_summary(manifest: DiscoveryManifest) -> str:
        latest = manifest.coverage_history[-1] if manifest.coverage_history else None
        coverage = f"{latest.weighted_score:.0%}" if latest else "not available"
        gaps = latest.gaps if latest else []
        lines = [
            "### Evidence Discovery",
            "",
            f"- Deep Research passes: {len(manifest.runs)} of 3",
            f"- Coverage: {coverage}",
            f"- Source leads: {len(manifest.source_leads)}",
            f"- Estimated cost: ${manifest.estimated_cost_usd:.2f}",
            f"- Stop reason: {manifest.convergence_reason or 'in progress'}",
            "- Status: discovered, not yet verified",
            "- Stored interaction notice: Deep Research uses stored Gemini interactions.",
        ]
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
            manifest_ok = False
            if manifest_artifact:
                manifest = DiscoveryManifest.model_validate(manifest_artifact.payload)
                manifest_ok = bool(manifest.source_leads) and not any(
                    run.status in {"failed", "cancelled", "timed_out", "incomplete"}
                    for run in manifest.runs
                )
            packet_ok = bool(
                packet_artifact
                and EvidencePacket.model_validate(packet_artifact.payload).verified
            )
            if not (manifest_ok and packet_ok):
                self.session.status = "evidence_required"
                self._persist(
                    self._event(
                        "evidence_verification_required",
                        "supervisor",
                        payload={"manifest_ok": manifest_ok, "packet_ok": packet_ok},
                    )
                )
                raise ValueError(
                    "Generation requires completed discovery and claim-level source "
                    "verification. Retry, or explicitly select the limited "
                    "exploratory workflow."
                )
        if (
            self.approval_profile == ApprovalProfile.ARTIFACT
            and self.pending_artifact_reviews
        ):
            raise ValueError(
                "Every specialist artifact must be approved before the stage bundle."
            )
        if artifact.stage == "reflect":
            governance_blockers = [
                review
                for item in self.session.artifacts
                if item.id in artifact.input_artifact_ids
                and item.agent == "ethics_safety_governance"
                and item.schema_name == "ReviewSet"
                for review in ReviewSet.model_validate(item.payload).reviews
                if review.fatal_flaws
            ]
            if governance_blockers:
                self.session.status = "governance_blocked"
                event = self._event(
                    "governance_blocked",
                    "supervisor",
                    payload={
                        "review_ids": [review.id for review in governance_blockers]
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

    def stop(self, *, actor: str = "researcher") -> None:
        if self.session.status != "active":
            return
        decision = HumanDecision(
            action=DecisionAction.STOP,
            stage=self.stage,
            actor=actor,
            session_version=self.session.version,
        )
        self.session.decisions.append(decision)
        self.session.status = "stopped_by_researcher"
        event = self._event(
            "session_stopped", actor, payload={"decision_id": decision.id}
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
