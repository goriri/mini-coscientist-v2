"""Web API for the durable, human-governed research workflow."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import secrets
import threading
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
from coscientist.models import ApprovalProfile, Artifact, ArtifactStatus
from coscientist.orchestration import CoScientistWorkflow
from coscientist.presentation import build_stage_presentation

router = APIRouter(prefix="/api/research", tags=["research-workflow"])
logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_STATE_DIR = Path(
    os.environ.get("COSCIENTIST_STATE_DIR", _PROJECT_DIR / ".coscientist")
)
_locks_guard = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}


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
    ]
    feedback: str = ""
    content: str = Field(default="", max_length=200000)
    artifact_id: str | None = None
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
    return {
        "id": session.id,
        "question": session.question,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "status": session.status,
        "stage": workflow.stage,
        "stage_number": min(session.current_stage + 1, len(workflow.workflow_stages)),
        "stage_count": len(workflow.workflow_stages),
        "approval_profile": session.approval_profile,
        "model": session.model,
        "language": session.language,
        "requires_human_approval": workflow.requires_human_approval,
        "literature_only": session.literature_only,
        "pending_draft": _artifact_summary(workflow.pending_draft, workflow),
        "pending_artifacts": [
            _artifact_summary(item, workflow)
            for item in workflow.pending_artifact_reviews
        ],
        "input_requirements": requirements,
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
        "operation": _operation(session.id),
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
    if not _ledger().claim_operation(session_id, owner, detail=details[kind]):
        return
    try:
        with _lock_for(session_id):
            workflow = _load(session_id)
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
                if workflow.approval_profile == ApprovalProfile.AUTO:
                    workflow.run_auto()
                else:
                    _draft_next_gate(workflow)
            elif kind == "revision":
                workflow.preview(feedback)
            else:
                _draft_next_gate(workflow)
        _set_operation(
            session_id, "completed", "The requested artifact is ready.", kind
        )
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
        _set_operation(
            session_id,
            "queued",
            f"Deep Research is still running; next status check in {delay} seconds.",
            "evidence",
        )
        enqueue_evidence_step(
            session_id,
            session_version=persisted.version,
            delay_seconds=delay,
        )
    except Exception as error:
        try:
            _set_operation(session_id, "failed", str(error), kind)
        except Exception:
            # A researcher may permanently delete a session while its worker
            # is finishing. The optimistic session write cannot recreate it.
            pass


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
    if (
        evidence_tasks_configured()
        and kind in {"next", "auto"}
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
            ledger=_ledger(),
        )
    except ValueError as error:
        # An unsupported research mode, model, or language is a rejected request,
        # not a server fault. It used to surface as a bare 500 with the reason
        # visible only in the Cloud Run log.
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


@router.get("/sessions/{session_id}")
def get_research_session(session_id: str, background_tasks: BackgroundTasks) -> dict:
    """Return the persisted approval and workflow state."""
    operation = _operation(session_id)
    if _ledger().requeue_expired_operation(session_id):
        background_tasks.add_task(
            _advance_in_background,
            session_id,
            kind=operation.get("kind", "next"),
            feedback="",
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


@router.delete("/sessions/{session_id}", status_code=204)
def delete_research_session(
    session_id: str,
    deletion_token: str | None = Header(default=None, alias="X-Session-Delete-Token"),
) -> None:
    """Permanently delete a bearer-owned research session."""
    if not deletion_token:
        raise HTTPException(status_code=401, detail="Deletion token required.")
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
                    _schedule_advance(session_id, background_tasks)
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
            elif request.action == "continue":
                if workflow.session.status == "evidence_required":
                    workflow.retry_evidence(actor=request.actor)
                    _schedule_advance(session_id, background_tasks)
                elif workflow.pending_draft is not None:
                    raise ValueError("A draft is already waiting for a decision.")
                else:
                    _schedule_advance(session_id, background_tasks)
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
            elif request.action == "exploratory_evidence":
                workflow.accept_exploratory_evidence(actor=request.actor)
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
