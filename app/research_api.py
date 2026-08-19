"""Web API for the durable, human-governed research workflow."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.app_utils.a2a import default_app_url
from app.evidence_tasks import configured as evidence_tasks_configured
from app.evidence_tasks import enqueue_evidence_step
from coscientist.agents import A2AProvider, DeterministicProvider
from coscientist.dossier import render_docx, render_pdf
from coscientist.evidence import EvidenceStillRunning
from coscientist.governance import governance_blockers
from coscientist.ledger import (
    ConcurrentSessionUpdate,
    PostgresResearchLedger,
    ResearchLedger,
)
from coscientist.model_catalog import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    LANGUAGE_CHOICES,
    LANGUAGE_CODES,
    MODEL_CHOICES,
    MODEL_IDS,
)
from coscientist.models import (
    ApprovalProfile,
    Artifact,
    ArtifactStatus,
    CandidatePopulation,
)
from coscientist.narrative import stage_name
from coscientist.orchestration import (
    WORKFLOW_STAGES,
    WORKFLOW_STAGES_V1,
    CoScientistWorkflow,
)
from coscientist.presentation import build_stage_presentation

router = APIRouter(prefix="/api/research", tags=["research-workflow"])
logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_STATE_DIR = Path(
    os.environ.get("COSCIENTIST_STATE_DIR", _PROJECT_DIR / ".coscientist")
)
_locks_guard = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}

OPERATION_LEASE_SECONDS = 300
"""How long a worker owns a session before it is presumed dead.

Short on purpose. A container that is torn down mid-stage should be noticed in
minutes, not in however long the longest stage can run -- and the heartbeat
below is what keeps a worker that is merely slow from being presumed dead.
"""

OPERATION_HEARTBEAT_SECONDS = 60
"""How often the heartbeat renews. Well inside the lease, so a missed beat is
survivable and only a stopped worker actually expires."""


class CreateResearchSession(BaseModel):
    question: str = Field(min_length=3, max_length=12000)
    approval_profile: ApprovalProfile = ApprovalProfile.MILESTONE
    research_mode: str | None = None
    # Validated here rather than on the Session, because this is the point at
    # which the value is chosen. An id nobody built a specialist tree for is a
    # 404 partway through a stage, which is a much worse way to learn about a
    # typo than a 422 on the request that made it.
    model: Literal[MODEL_IDS] = DEFAULT_MODEL  # type: ignore[valid-type]
    language: Literal[LANGUAGE_CODES] = DEFAULT_LANGUAGE  # type: ignore[valid-type]
    # Off unless asked for. The extra gate is only useful to a caller who is
    # going to read what is behind it, and a script polling this API to
    # completion would simply park at evidence forever. The web launcher asks
    # for it explicitly; nothing else inherits it.
    evidence_review: bool = False
    # A run that exercises the pipeline instead of proposing research. It waives
    # its own governance gate -- in writing, on the record, and printed in the
    # report -- so a build that has to reach the report stage does not park at
    # reflect waiting for a person to sign for a hazard nobody intends to go
    # near. Off unless asked for: the flag changes what the output may be taken
    # for, so it cannot be inferred, only declared.
    rehearsal: bool = False
    # An earlier run of this same question whose scope and evidence base this one
    # starts from. Gathering the corpus is the long half of a run -- eight Deep
    # Research passes, an hour, twenty-four dollars -- and asking the same question
    # again buys them a second time to arrive at the same place.
    seed_evidence_from: str = ""


class ResearchDecision(BaseModel):
    action: Literal[
        "accept",
        "revise",
        "stop",
        "approve_artifact",
        "literature_only",
        "exploratory_evidence",
        "provide_input",
        "edit",
        "continue",
        "refine_section",
        # The web had no answer to a governance block at all. The reflect stage
        # sets that status by design, the CLI and the TUI can clear it, and a
        # researcher in the browser could only watch: Accept returns silently,
        # the status never changes, and the session is finished. These two are
        # the same two answers the TUI offers, one finding at a time.
        "withdraw_hypothesis",
        "override_governance",
    ]
    feedback: str = ""
    content: str = Field(default="", max_length=200000)
    artifact_id: str | None = None
    review_id: str | None = None
    candidate_id: str | None = None
    section: str | None = None
    input_type: str | None = None
    input_reference: str | None = None
    actor: str = Field(default="web_researcher", max_length=120)


@functools.cache
def _ledger() -> ResearchLedger | PostgresResearchLedger:
    if database_url := os.environ.get("COSCIENTIST_DATABASE_URL"):
        return PostgresResearchLedger(database_url)
    if connection_name := os.environ.get("CLOUD_SQL_CONNECTION_NAME"):
        user = quote_plus(os.environ.get("DATABASE_USER", "coscientist_app"))
        password = quote_plus(os.environ["DATABASE_PASSWORD"])
        database = quote_plus(os.environ.get("DATABASE_NAME", "coscientist"))
        socket = quote_plus(f"/cloudsql/{connection_name}")
        return PostgresResearchLedger(
            f"postgresql://{user}:{password}@/{database}?host={socket}"
        )
    return ResearchLedger(_STATE_DIR / "research_workflows.db")


def _provider(model: str = DEFAULT_MODEL):
    if os.getenv("INTEGRATION_TEST", "").upper() == "TRUE":
        return DeterministicProvider()
    # Same base URL the specialist cards advertise, so the self-call and the
    # published card can never drift onto different ports.
    return A2AProvider(default_app_url(), model=model)


def _lock_for(session_id: str) -> threading.Lock:
    with _locks_guard:
        return _session_locks.setdefault(session_id, threading.Lock())


def _set_operation(
    session_id: str, status: str, detail: str = "", kind: str = "generation"
) -> None:
    _ledger().set_operation(session_id, status, detail, kind)


def _operation(session_id: str) -> dict[str, str]:
    return _ledger().operation(session_id)


def _reported_status(status: str, operation: dict) -> str:
    """What the run is, once the worker that was driving it has stopped.

    A worker that dies records the reason against the operation and leaves the
    session alone, because the session is still exactly where the failure left
    it and a retry has to be able to pick it up from there. Nothing then said
    so: the run stayed ``active`` and the workspace read "Workflow active" over
    a run that had ended hours earlier. Four runs on the live deployment sat
    like that -- three of them killed by a specialist closing its event stream
    without answering -- and the landing screen, which now opens whatever is
    running, opened one of them and watched a spinner that would never move.

    Read here rather than written back, so a retry that requeues the operation
    puts the run back to active by itself, and so the four already stranded are
    told the truth the next time they are asked without anything editing them.
    """
    if status == "active" and operation.get("status") == "failed":
        return "failed"
    return status


def _artifact_summary(
    artifact: Artifact | None, workflow: CoScientistWorkflow | None = None
) -> dict | None:
    if artifact is None:
        return None
    summary = {
        "id": artifact.id,
        "agent": artifact.agent,
        "stage": artifact.stage,
        "version": artifact.version,
        "status": artifact.status,
        "content": artifact.content,
    }
    if workflow is not None:
        summary["presentation"] = build_stage_presentation(
            workflow.session, artifact.stage
        )
    return summary


def _select_stage_preview(workflow: CoScientistWorkflow, stage: str) -> Artifact | None:
    bundles = [
        artifact
        for artifact in workflow.session.artifacts
        if artifact.stage == stage and artifact.artifact_type == "stage_bundle"
    ]
    if not bundles:
        return None
    if not workflow.done and workflow.stage == stage:
        pending = workflow.pending_draft
        if pending is not None:
            return pending
    return next(
        (
            artifact
            for artifact in reversed(bundles)
            if artifact.status == ArtifactStatus.ACCEPTED
        ),
        bundles[-1],
    )


def _stage_preview_metadata(workflow: CoScientistWorkflow) -> list[dict]:
    current_index = workflow.session.current_stage
    return [
        {
            "stage": stage,
            "available": (artifact := _select_stage_preview(workflow, stage))
            is not None,
            "artifact_id": artifact.id if artifact else None,
            "version": artifact.version if artifact else None,
            "status": artifact.status if artifact else None,
            "is_current": not workflow.done and workflow.stage == stage,
            "is_completed": workflow.done or index < current_index,
        }
        for index, stage in enumerate(workflow.workflow_stages)
    ]


def _candidate_titles(workflow: CoScientistWorkflow) -> dict[str, str]:
    """Titles for every candidate the session has ever held.

    Every population, not only the live one: withdrawing a hypothesis rewrites
    the population without it, and a finding that has just been answered still
    has to be readable in the card that answered it.
    """
    titles: dict[str, str] = {}
    for artifact in workflow.session.artifacts:
        if artifact.schema_name != "CandidatePopulation" or not artifact.payload:
            continue
        for item in CandidatePopulation.model_validate(artifact.payload).candidates:
            titles[item.id] = item.title or item.claim
    return titles


def _governance_blockers(workflow: CoScientistWorkflow) -> list[dict]:
    """Every fatal safety finding at a live governance block, answered or not.

    The flaw itself travels with the blocker, not just its id. A researcher is
    being asked to either drop a hypothesis or put their name to keeping one
    that a reviewer called dangerous, and that decision cannot be made from a
    review identifier.

    Answered findings are carried too, with the answer attached, because only
    the open ones block but all of them are what the researcher is working
    through. Sending the open ones alone is what made the web card unusable: a
    run with four findings tore the card down and rebuilt it as a new card of
    three the moment the first was withdrawn, with no sign of the one just
    answered and every part-typed reason in the others lost. Once the last
    finding is answered the block is over and the list empties, so a cleared
    gate does not carry its history into the next stage's card.
    """
    findings = governance_blockers(workflow.session)
    answered = {
        item.review_id: item for item in workflow.session.governance_adjudications
    }
    if not findings or all(item.review_id in answered for item in findings):
        return []
    titles = _candidate_titles(workflow)
    return [
        {
            "review_id": item.review_id,
            "candidate_id": item.candidate_id,
            "candidate_title": titles.get(item.candidate_id, item.candidate_id),
            "reviewer": item.review.reviewer,
            "fatal_flaws": list(item.review.fatal_flaws),
            "objections": list(item.review.objections),
            "resolution": (
                {
                    "action": decided.resolution,
                    "actor": decided.adjudicator,
                    "reason": decided.justification,
                    "decided_at": decided.created_at,
                }
                if (decided := answered.get(item.review_id))
                else None
            ),
        }
        for item in findings
    ]


def _snapshot(workflow: CoScientistWorkflow) -> dict:
    session = workflow.session
    discovery_artifact = next(
        (
            item
            for item in reversed(session.artifacts)
            if item.schema_name == "DiscoveryManifest" and item.payload
        ),
        None,
    )
    requirements = [
        {
            "id": item.id,
            "input_type": item.input_type,
            "description": item.description,
            "reason": item.reason,
            "blocking": item.blocking,
            "permitted_fallback": item.permitted_fallback,
            "status": item.status,
        }
        for item in session.input_requirements
    ]
    decisions = [
        {
            "id": item.id,
            "action": item.action,
            "stage": item.stage,
            "actor": item.actor,
            "automatic": item.automatic,
            "feedback": item.feedback,
        }
        for item in session.decisions[-12:]
    ]
    operation = _operation(session.id)
    return {
        "id": session.id,
        "question": session.question,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "status": _reported_status(session.status, operation),
        "stage": workflow.stage,
        "stage_number": min(session.current_stage + 1, len(workflow.workflow_stages)),
        "stage_count": len(workflow.workflow_stages),
        "approval_profile": session.approval_profile,
        "model": session.model,
        "language": session.language,
        "requires_human_approval": workflow.requires_human_approval,
        # What the run settled on, not what was asked for: an auto run drops the
        # gate, and the launcher needs to be able to say so rather than promise
        # a stop that never comes.
        "evidence_review": session.evidence_review,
        # Sent on every snapshot so the badge is on the page from the first poll.
        # A reader who opens a run halfway through has no other way to tell a
        # rehearsal from a proposal until the report exists, and by then the
        # distinction has stopped being useful.
        "rehearsal": session.rehearsal,
        # A forked run's evidence card reports the search that built the corpus --
        # seven passes, ninety leads, twenty-one dollars -- and none of it was this
        # run's. The report says so, but the report is the last thing to exist; a
        # person watching the run needs it from the stage the corpus shows up in.
        "seeded_evidence_from": session.seeded_evidence_from,
        "literature_only": session.literature_only,
        "pending_draft": _artifact_summary(workflow.pending_draft, workflow),
        "pending_artifacts": [
            _artifact_summary(item, workflow)
            for item in workflow.pending_artifact_reviews
        ],
        "input_requirements": requirements,
        "governance_blockers": _governance_blockers(workflow),
        "decisions": decisions,
        "task_summary": {
            "total": len(session.tasks),
            "completed": sum(item.state == "completed" for item in session.tasks),
            "failed": sum(item.state == "failed" for item in session.tasks),
        },
        "evidence_progress": (
            build_stage_presentation(session, "evidence")
            if discovery_artifact is not None
            else None
        ),
        "stage_previews": _stage_preview_metadata(workflow),
        "report_available": workflow.done,
        "operation": operation,
        "report": workflow.render_report() if workflow.done else None,
        "report_presentation": (
            build_stage_presentation(session, "meta_review") if workflow.done else None
        ),
        "report_exports": (
            [
                {
                    "format": report_format,
                    "label": label,
                    "filename": workflow.report_filename().removesuffix(".md") + suffix,
                    "url": (
                        f"/api/research/sessions/{session.id}/report/{report_format}"
                    ),
                }
                for report_format, label, suffix in (
                    ("docx", "Google Docs (.docx)", ".docx"),
                    ("pdf", "PDF", ".pdf"),
                    ("md", "Markdown", ".md"),
                )
            ]
            if workflow.done
            else []
        ),
    }


def _load(session_id: str) -> CoScientistWorkflow:
    # The provider has to exist before the session that names its model can be
    # read, so it is built on the default and the workflow rebinds it from the
    # loaded session. See ``bind_provider_model`` in coscientist.agents.
    try:
        return CoScientistWorkflow.load_from_ledger(
            session_id, _ledger(), provider=_provider()
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def _resume_kind(workflow: CoScientistWorkflow) -> str:
    """How a run that was interrupted should be started up again.

    An auto run answers its own gates, and the worker that does that is the one
    started with kind "auto". Every resumption point below used the default
    instead, which drafts the next gate and stops -- so an auto run that was
    interrupted by a governance block never moved again once the block was
    answered. Three production runs parked at reflect that way: status active,
    profile auto, a draft waiting and nobody left who would accept it.
    """
    return "auto" if workflow.approval_profile == ApprovalProfile.AUTO else "next"


def _draft_next_gate(workflow: CoScientistWorkflow) -> None:
    if workflow.done or workflow.session.status != "active":
        return
    if workflow.approval_profile == ApprovalProfile.MILESTONE:
        workflow.advance_to_human_gate()
    if not workflow.done and workflow.pending_draft is None:
        workflow.preview()


def _advance_in_background(session_id: str, *, kind: str, feedback: str = "") -> None:
    details = {
        "initial": "The goal specialist is framing the first research gate.",
        "revision": "The specialist is generating a revised artifact.",
        "auto": "The autonomous planning workflow is generating the dossier.",
        "next": "Specialists are preparing the next research gate.",
        "evidence": "Deep Research is starting or polling a stored interaction.",
    }
    owner = f"worker-{secrets.token_hex(8)}"
    if not _ledger().claim_operation(
        session_id, owner, detail=details[kind], lease_seconds=OPERATION_LEASE_SECONDS
    ):
        return
    stop_heartbeat = threading.Event()
    threading.Thread(
        target=_hold_operation_lease,
        args=(session_id, owner, stop_heartbeat),
        name=f"lease-{session_id}",
        daemon=True,
    ).start()
    try:
        _run_advance(session_id, owner, kind=kind, feedback=feedback, details=details)
    finally:
        stop_heartbeat.set()


def _progress_writer(session_id: str, owner: str, kind: str):
    """Let a stage rewrite the line the workspace is showing while it runs.

    Through ``renew_operation`` and not ``set_operation``: the second clears the
    lease columns, so a stage narrating its own progress would hand its session
    to the expiry sweep and have a second worker started beside it. The first
    holds the lease it is already holding and replaces only the sentence -- the
    heartbeat's write with a message attached.

    The lease going away means this worker is no longer the one on the session,
    and the right thing to do about a sentence nobody asked us for is nothing.
    """

    def write(detail: str) -> None:
        _ledger().renew_operation(
            session_id,
            owner,
            detail=detail,
            lease_seconds=OPERATION_LEASE_SECONDS,
        )
        logger.info("%s: %s (%s)", session_id, detail, kind)

    return write


def _hold_operation_lease(session_id: str, owner: str, stop: threading.Event) -> None:
    """Renew the lease for as long as the worker is still on the session.

    A stage is one long call: discovery polls every Deep Research pass to
    completion inside it, which is many minutes against a five-minute lease.
    Without this the expiry sweep declared a working worker dead and started a
    second one beside it every five minutes, so the evidence stage restarted
    forever instead of finishing -- attempt five, twenty-five minutes in, still
    at pass seven.

    The message is left alone. What the researcher is being told is whatever the
    worker last set, and a heartbeat has nothing to add to it.
    """
    while not stop.wait(OPERATION_HEARTBEAT_SECONDS):
        try:
            if not _ledger().renew_operation(
                session_id, owner, lease_seconds=OPERATION_LEASE_SECONDS
            ):
                # The lease is somebody else's now, or the session is gone.
                return
        except Exception:
            logger.exception("Could not renew the operation lease for %s", session_id)
            return


def _serving_one_task_at_a_time() -> bool:
    """Whether this process is the queue's worker rather than the web service.

    The same switch the discovery controller reads to poll once and return, so
    a process cannot be bounded on the poll loop and unbounded on the stages
    around it.
    """
    return os.environ.get("EVIDENCE_TASK_STEP_MODE", "false").lower() == "true"


def _advance_one_stage(workflow: CoScientistWorkflow) -> None:
    """Accept one stage, leaving the next one for the next task.

    The evidence gate is let through the way ``run_auto`` lets it through: a
    run whose evidence base is too thin to hypothesise from stops for a person,
    and that is a decision, not a failure of this worker.
    """
    try:
        workflow.accept(workflow.preview(), automatic=True)
    except ValueError:
        if workflow.session.status != "evidence_required":
            raise


def _run_advance(
    session_id: str,
    owner: str,
    *,
    kind: str,
    feedback: str,
    details: dict[str, str],
) -> None:
    # Deep Research answers in minutes, and each poll that comes back "still
    # running" schedules the next one. Where Cloud Tasks is configured that next
    # poll is a task and this worker returns; where it is not, the worker waits
    # here and polls again, which is what the loop is for.
    while True:
        try:
            with _lock_for(session_id):
                workflow = _load(session_id)
                workflow.progress = _progress_writer(session_id, owner, kind)
                if kind == "auto" and evidence_tasks_configured():
                    while (
                        not workflow.done
                        and workflow.stage != "evidence"
                        and workflow.session.status == "active"
                    ):
                        workflow.accept(workflow.preview(), automatic=True)
                    if workflow.stage == "evidence":
                        _set_operation(
                            session_id,
                            "queued",
                            "Deep Research pass 1 is queued.",
                            "evidence",
                        )
                        enqueue_evidence_step(
                            session_id,
                            session_version=workflow.session.version,
                            delay_seconds=0,
                        )
                        return
                elif kind == "auto":
                    workflow.run_auto()
                elif kind == "evidence":
                    if workflow.approval_profile != ApprovalProfile.AUTO:
                        _draft_next_gate(workflow)
                    elif _serving_one_task_at_a_time():
                        # A task is one stage, and the stage after it is another
                        # task. ``run_auto`` here would carry the whole rest of
                        # the pipeline -- generate, five reviewers, a tournament,
                        # evolution, a meta-review -- inside a single request
                        # that Cloud Run cuts off at three hundred seconds, and
                        # the retry then lands on a lease the killed instance
                        # holds for five more minutes. The poll loop below is
                        # already bounded this way; this is the same bound
                        # applied to the stages that follow it.
                        _advance_one_stage(workflow)
                        if not workflow.done and workflow.session.status == "active":
                            _set_operation(
                                session_id,
                                "queued",
                                f"Queued: {stage_name(workflow.stage)}.",
                                "evidence",
                            )
                            enqueue_evidence_step(
                                session_id,
                                session_version=workflow.session.version,
                                delay_seconds=0,
                            )
                            return
                    else:
                        workflow.run_auto()
                elif kind == "revision":
                    workflow.preview(feedback)
                else:
                    _draft_next_gate(workflow)
            _set_operation(
                session_id, "completed", "The requested artifact is ready.", kind
            )
            return
        except EvidenceStillRunning:
            persisted = _ledger().load(session_id)
            discovery = next(
                (
                    item
                    for item in reversed(persisted.artifacts)
                    if item.schema_name == "DiscoveryManifest" and item.payload
                ),
                None,
            )
            polls = 0
            if discovery:
                runs = discovery.payload.get("runs") or []
                if runs:
                    polls = int(runs[-1].get("poll_count", 0))
            delay = min(60, 15 * (2 ** min(2, polls // 5)))
            waiting = (
                f"Deep Research is still running; next status check in {delay} seconds."
            )
            if evidence_tasks_configured():
                _set_operation(session_id, "queued", waiting, "evidence")
                enqueue_evidence_step(
                    session_id,
                    session_version=persisted.version,
                    delay_seconds=delay,
                )
                return
            # Without a task queue there is nobody to hand the next poll to.
            # Handing it to ``enqueue_evidence_step`` anyway raised inside this
            # handler and killed the worker with the operation left queued, so
            # the stage only advanced when a lease expiry happened to restart
            # it: one poll every five minutes, and a session that never
            # finished discovery.
            #
            # The lease is renewed rather than released, and the operation stays
            # ``running``, so an instance that dies mid-wait is still recovered
            # by ``requeue_expired_operation``. A renewal that fails means the
            # lease was taken away, and this worker stops rather than writing to
            # a session another one now owns.
            if not _ledger().renew_operation(
                session_id,
                owner,
                detail=waiting,
                lease_seconds=OPERATION_LEASE_SECONDS,
            ):
                return
            time.sleep(delay)
        except Exception as error:
            # Logged before it is recorded. What the researcher is shown is
            # ``str(error)``, and "list index out of range" with no traceback
            # anywhere names neither the stage nor the line -- a real production
            # failure took an extra deploy to find for want of these two lines.
            logger.exception(
                "The %s worker for %s failed", kind, session_id, exc_info=error
            )
            try:
                _set_operation(session_id, "failed", str(error), kind)
            except Exception:
                # A researcher may permanently delete a session while its worker
                # is finishing. The optimistic session write cannot recreate it.
                pass
            return


def _hand_off(
    session_id: str,
    background_tasks: BackgroundTasks,
    *,
    kind: str,
    feedback: str = "",
) -> None:
    """Give the work to the queue where there is one, and to this process where
    there is not.

    Every route into a stage goes through here, recovery included. The recovery
    route used to add the background task directly, and a background task is the
    thing the queue exists to replace: it outlives the request that started it,
    so the instance serving it can be reclaimed mid-stage, which is what leaves
    an evidence stage to be recovered in the first place. A live run went
    straight back onto that path -- the sweep declared the interrupted worker
    dead, started attempt three inside the main service, and the queue standing
    ready beside it was not asked.
    """
    if (
        evidence_tasks_configured()
        and kind in {"next", "auto", "evidence"}
        and not feedback
        and _load(session_id).stage == "evidence"
    ):
        enqueue_evidence_step(
            session_id,
            session_version=_load(session_id).session.version,
            delay_seconds=0,
        )
        return
    background_tasks.add_task(
        _advance_in_background,
        session_id,
        kind=kind,
        feedback=feedback,
    )


def _schedule_advance(
    session_id: str,
    background_tasks: BackgroundTasks,
    *,
    kind: str = "next",
    feedback: str = "",
) -> None:
    operation = _operation(session_id)
    if operation["status"] in {"queued", "running"}:
        raise ValueError("The next research gate is already being prepared.")
    _set_operation(session_id, "queued", "Waiting for a workflow worker.", kind)
    _hand_off(session_id, background_tasks, kind=kind, feedback=feedback)


@router.get("/options")
def research_options() -> dict:
    """What a new run may be configured with.

    The form reads its choices from here rather than hard-coding them, so the
    allowlist the server validates against and the list a researcher picks from
    cannot disagree -- a browser cache would otherwise keep offering a model the
    server has since retired.
    """
    return {
        "models": [
            {
                "id": choice.id,
                "label": choice.label,
                "note": choice.note,
                "default": choice.id == DEFAULT_MODEL,
            }
            for choice in MODEL_CHOICES
        ],
        "languages": [
            {
                "code": choice.code,
                "label": choice.label,
                "endonym": choice.endonym,
                "default": choice.code == DEFAULT_LANGUAGE,
            }
            for choice in LANGUAGE_CHOICES
        ],
    }


@router.post("/sessions", status_code=201)
def create_research_session(
    request: CreateResearchSession, background_tasks: BackgroundTasks
) -> dict:
    """Create and draft the first governed research stage."""
    try:
        workflow = CoScientistWorkflow(
            request.question,
            provider=_provider(request.model),
            approval_profile=request.approval_profile,
            research_mode=request.research_mode,
            model=request.model,
            language=request.language,
            workflow_version=int(os.environ.get("EVIDENCE_PIPELINE_VERSION", "2")),
            evidence_review=request.evidence_review,
            rehearsal=request.rehearsal,
            ledger=_ledger(),
        )
    except ValueError as error:
        # An unsupported research mode, model, or language is a rejected request,
        # not a server fault. It used to surface as a bare 500 with the reason
        # visible only in the Cloud Run log.
        raise HTTPException(status_code=400, detail=str(error)) from error
    if request.seed_evidence_from:
        # The 404 and the 400 say different things: an id nobody holds is a typo in
        # the field, and a run that cannot be forked is a run the caller has to pick
        # differently. Reported as one status they were indistinguishable in the
        # launcher, which is where the id is pasted in.
        try:
            source = _ledger().load(request.seed_evidence_from)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=f"No session {request.seed_evidence_from} to fork.",
            ) from error
        try:
            workflow.seed_evidence_from(source)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    kind = "auto" if workflow.approval_profile == ApprovalProfile.AUTO else "initial"
    deletion_token = secrets.token_urlsafe(32)
    _ledger().set_delete_token_hash(
        workflow.session.id, hashlib.sha256(deletion_token.encode()).hexdigest()
    )
    _schedule_advance(workflow.session.id, background_tasks, kind=kind)
    snapshot = _snapshot(workflow)
    snapshot["deletion_token"] = deletion_token
    return snapshot


@router.get("/sessions")
def list_research_sessions(limit: int = 50) -> dict:
    """Every run on this server -- running, blocked and finished -- to anyone.

    The history panel was a browser's own localStorage until now, so a visitor
    arriving at the service saw an empty page however much was running on it,
    and a researcher who opened the site on a second machine could not reach a
    single run they had started on the first.

    Nothing here is loaded through ``_load``: that builds a workflow and, for a
    finished run, renders its whole dossier. This route answers from the six
    scalars the ledger pulls out in SQL, so listing fifty runs costs about what
    listing one used to.
    """
    ledger = _ledger()
    listing = []
    for entry in ledger.recent_sessions(min(max(1, limit), 200)):
        operation = ledger.operation(entry["id"])
        stages = (
            WORKFLOW_STAGES if entry["workflow_version"] >= 2 else WORKFLOW_STAGES_V1
        )
        position = entry["current_stage"]
        done = position >= len(stages)
        listing.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "status": _reported_status(entry["status"], operation),
                "stage": "report" if done else stages[position],
                "stage_number": min(position + 1, len(stages)),
                "stage_count": len(stages),
                "report_available": done,
                "created_at": entry["created_at"],
                "updated_at": entry["updated_at"],
                # What the run is waiting on, which is the difference between a
                # run nobody has to touch and one parked at a gate. The panel
                # showed neither before, because it had never heard of the run.
                "operation": operation,
            }
        )
    return {"sessions": listing}


@router.get("/sessions/{session_id}")
def get_research_session(session_id: str, background_tasks: BackgroundTasks) -> dict:
    """Return the persisted approval and workflow state."""
    operation = _operation(session_id)
    if _ledger().requeue_expired_operation(session_id):
        _hand_off(
            session_id,
            background_tasks,
            kind=operation.get("kind", "next"),
        )
    return _snapshot(_load(session_id))


@router.get("/sessions/{session_id}/stages/{stage}")
def get_stage_preview(session_id: str, stage: str) -> dict:
    """Return one read-only Supervisor stage bundle without mutating workflow state."""
    workflow = _load(session_id)
    if stage not in workflow.workflow_stages:
        raise HTTPException(status_code=404, detail=f"Unknown research stage: {stage}")
    artifact = _select_stage_preview(workflow, stage)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stage preview is available for {stage}.",
        )
    return {
        "session_id": session_id,
        "stage": stage,
        "artifact_id": artifact.id,
        "version": artifact.version,
        "status": artifact.status,
        "producer": artifact.agent,
        "producer_model": artifact.producer_model,
        "created_at": artifact.created_at,
        "feedback": artifact.feedback,
        "content": artifact.content,
        "presentation": build_stage_presentation(workflow.session, stage),
        "read_only": True,
    }


@router.get("/sessions/{session_id}/report/{report_format}")
def download_research_report(session_id: str, report_format: str) -> Response:
    """Download a completed dossier in Markdown, PDF, or editable DOCX."""
    workflow = _load(session_id)
    if not workflow.done:
        raise HTTPException(
            status_code=409,
            detail="The final dossier is available after Meta-review is accepted.",
        )
    content = workflow.render_report()
    basename = workflow.report_filename().removesuffix(".md")
    if report_format == "md":
        body = content.encode("utf-8")
        media_type = "text/markdown; charset=utf-8"
        suffix = ".md"
    elif report_format == "pdf":
        body = render_pdf(content)
        media_type = "application/pdf"
        suffix = ".pdf"
    elif report_format == "docx":
        body = render_docx(content)
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        suffix = ".docx"
    else:
        raise HTTPException(status_code=404, detail="Unknown report format.")
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{basename}{suffix}"',
            "Cache-Control": "private, no-store",
        },
    )


def _is_operator(token: str) -> bool:
    """Whether this credential is the deployment's own, rather than a session's.

    Set ``COSCIENTIST_ADMIN_TOKEN`` on the service and whoever holds it may
    delete any run on it. Unset -- which is every local run and every test --
    there is no operator and the per-session token is the only key there is.
    """
    admin = os.environ.get("COSCIENTIST_ADMIN_TOKEN", "")
    return bool(admin) and secrets.compare_digest(token, admin)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_research_session(
    session_id: str,
    deletion_token: str | None = Header(default=None, alias="X-Session-Delete-Token"),
) -> None:
    """Permanently delete a research session, by its own token or the operator's.

    The per-session token is handed out once, to the browser that created the
    run, and nothing hands it out again. So a run started on another machine,
    from the command line, or before this service issued tokens showed no
    delete button at all -- the reason a live deployment reached sixty-nine
    sessions with no way to clear any of them.

    A run that is not here answers 404 whatever credential is offered. The
    ledger reports "no such session" and "wrong token" as the same false, and
    reporting both as 403 stranded rows the browser kept for runs the server
    had already dropped: the page called them undeletable and offered the
    button again, so one row was refused eight times in ninety seconds. Which
    sessions exist is on the public listing anyway, so saying so tells a caller
    nothing it could not already read.
    """
    if not deletion_token:
        raise HTTPException(status_code=401, detail="Deletion token required.")
    try:
        _ledger().load(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="No such session.") from error
    if _is_operator(deletion_token):
        if not _ledger().delete_session(session_id, "", administrative=True):
            raise HTTPException(status_code=404, detail="No such session.")
        return
    token_hash = hashlib.sha256(deletion_token.encode()).hexdigest()
    if not _ledger().delete_session(session_id, token_hash):
        raise HTTPException(status_code=403, detail="Invalid deletion token.")


@router.get("/health/storage")
def storage_health() -> dict:
    """Report storage readiness without exposing connection details."""
    try:
        ready = _ledger().healthcheck()
    except Exception:
        logger.exception("Persistent research storage health check failed.")
        ready = False
    if not ready:
        raise HTTPException(status_code=503, detail="Persistent storage unavailable.")
    return {
        "ready": True,
        "backend": "postgresql"
        if os.environ.get("COSCIENTIST_DATABASE_URL")
        or os.environ.get("CLOUD_SQL_CONNECTION_NAME")
        else "sqlite",
    }


@router.post("/sessions/{session_id}/decisions")
def decide_research_session(
    session_id: str,
    request: ResearchDecision,
    background_tasks: BackgroundTasks,
) -> dict:
    """Apply a researcher decision, then stop at the next configured gate."""
    with _lock_for(session_id):
        workflow = _load(session_id)
        try:
            if request.action == "accept":
                draft = workflow.pending_draft
                if draft is None:
                    raise ValueError("There is no current stage draft to accept.")
                workflow.accept(draft, actor=request.actor, automatic=False)
                if not workflow.done and workflow.session.status == "active":
                    _schedule_advance(
                        session_id, background_tasks, kind=_resume_kind(workflow)
                    )
            elif request.action == "revise":
                workflow.request_revision(request.feedback, actor=request.actor)
                _schedule_advance(
                    session_id,
                    background_tasks,
                    kind="revision",
                    feedback=request.feedback,
                )
            elif request.action == "edit":
                workflow.edit_draft(request.content, actor=request.actor)
            elif request.action == "refine_section":
                if not request.candidate_id or not request.section:
                    raise ValueError(
                        "candidate_id and section are required for refine_section."
                    )
                workflow.refine_section(
                    request.candidate_id,
                    request.section,
                    request.feedback,
                    actor=request.actor,
                )
            elif request.action == "continue":
                if workflow.session.status == "evidence_required":
                    workflow.retry_evidence(actor=request.actor)
                    _schedule_advance(
                        session_id, background_tasks, kind=_resume_kind(workflow)
                    )
                elif (
                    workflow.pending_draft is not None
                    and workflow.approval_profile != ApprovalProfile.AUTO
                ):
                    # On every other profile a waiting draft belongs to whoever
                    # is going to decide it, and resuming underneath them would
                    # take the decision away. An unattended run has no such
                    # person: its draft is the auto worker's own next step, and
                    # refusing to resume is what kept three live sessions parked
                    # at reflect with a draft nobody was ever going to accept.
                    raise ValueError("A draft is already waiting for a decision.")
                else:
                    _schedule_advance(
                        session_id, background_tasks, kind=_resume_kind(workflow)
                    )
            elif request.action == "stop":
                workflow.stop(actor=request.actor)
            elif request.action == "approve_artifact":
                artifact = next(
                    (
                        item
                        for item in workflow.pending_artifact_reviews
                        if item.id == request.artifact_id
                    ),
                    None,
                )
                if artifact is None:
                    raise ValueError("The specialist artifact is stale or unavailable.")
                workflow.approve_artifact(artifact, actor=request.actor)
            elif request.action == "literature_only":
                workflow.accept_literature_only(actor=request.actor)
                _schedule_advance(
                    session_id, background_tasks, kind=_resume_kind(workflow)
                )
            elif request.action == "exploratory_evidence":
                workflow.accept_exploratory_evidence(actor=request.actor)
                _schedule_advance(
                    session_id, background_tasks, kind=_resume_kind(workflow)
                )
            elif request.action in {"withdraw_hypothesis", "override_governance"}:
                if not request.review_id:
                    raise ValueError(
                        "review_id is required to answer a governance finding."
                    )
                justification = request.feedback.strip()
                if not justification:
                    raise ValueError(
                        "A written justification is recorded with every "
                        "governance decision."
                    )
                # The default web actor is "web_researcher". An override that
                # keeps a hypothesis a reviewer called fatal has to name the
                # person who kept it, so anonymity is refused here rather than
                # written into the dossier.
                if request.actor.strip() in {"", "web_researcher"}:
                    raise ValueError(
                        "Name the person adjudicating this finding; the name is "
                        "recorded in the dossier beside the flaw."
                    )
                workflow.adjudicate_governance(
                    request.review_id,
                    "withdraw"
                    if request.action == "withdraw_hypothesis"
                    else "override",
                    adjudicator=request.actor.strip(),
                    justification=justification,
                )
                # Answering the last finding puts the session back exactly where
                # the block interrupted it: a reflect draft nobody has accepted,
                # on a profile that was going to accept it automatically. Left
                # alone it sits there, which is a quieter dead end than the one
                # this action exists to clear. The other two waivers resume the
                # same way.
                if workflow.session.status == "active":
                    _schedule_advance(
                        session_id, background_tasks, kind=_resume_kind(workflow)
                    )
            elif request.action == "provide_input":
                if not request.input_type or not request.input_reference:
                    raise ValueError("Input type and reference are required.")
                workflow.provide_input(
                    request.input_type,
                    request.input_reference,
                    actor=request.actor,
                )
            return _snapshot(workflow)
        except ConcurrentSessionUpdate as error:
            raise HTTPException(
                status_code=409,
                detail="This workflow changed in another browser. Refresh and retry.",
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
