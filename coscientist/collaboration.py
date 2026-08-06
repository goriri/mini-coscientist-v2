"""Async specialist task bus used by the local Supervisor.

The network A2A surface is supplied by the generated agents-cli FastAPI runtime.
This module mirrors the same task/card/artifact contract for offline execution,
so tests and the TUI exercise asynchronous purpose-specific collaboration
without pretending that an in-process call crossed a network.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .agents import DeterministicProvider, Provider, Specialist
from .models import (
    AgentCard,
    Artifact,
    Session,
    TaskRecord,
    TaskState,
    utc_now,
)


@dataclass(frozen=True)
class TaskResult:
    task: TaskRecord
    artifact: Artifact


class SpecialistService:
    def __init__(self, specialist: Specialist, provider: Provider):
        self.specialist = specialist
        self.provider = provider
        if specialist.role == "evidence_discovery":
            tools = ["google_search"]
        elif specialist.role == "source_verification":
            tools = ["fetch_source_document"]
        else:
            tools = []
        self.card = AgentCard(
            name=specialist.role,
            purpose=specialist.instruction,
            stage=specialist.stage,
            tools=tools,
        )

    async def execute(
        self,
        session: Session,
        *,
        feedback: str,
        idempotency_key: str,
    ) -> TaskResult:
        task = TaskRecord(
            context_id=session.context_id,
            stage=self.specialist.stage,
            agent=self.specialist.role,
            idempotency_key=idempotency_key,
            state=TaskState.WORKING,
            input_artifact_ids=[
                artifact.id
                for artifact in session.artifacts
                if artifact.status == "accepted"
            ][-6:],
        )
        try:
            if isinstance(self.provider, DeterministicProvider):
                # The offline provider is immediate and thread-safe execution adds
                # shutdown latency on some Python 3.13 event-loop implementations.
                artifact = self.specialist.run(session, self.provider, feedback)
            else:
                artifact = await asyncio.to_thread(
                    self.specialist.run, session, self.provider, feedback
                )
            artifact.artifact_type = "specialist_output"
            artifact.input_artifact_ids = list(task.input_artifact_ids)
            task.state = TaskState.COMPLETED
            task.output_artifact_id = artifact.id
        except Exception as exc:
            task.state = TaskState.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            task.updated_at = utc_now()
            raise
        task.updated_at = utc_now()
        return TaskResult(task=task, artifact=artifact)


class LocalA2ATaskBus:
    """Bounded asynchronous dispatcher for the project-local provider."""

    def __init__(
        self,
        specialists: tuple[Specialist, ...],
        provider: Provider,
        *,
        max_concurrency: int = 4,
    ):
        self.services = {
            specialist.role: SpecialistService(specialist, provider)
            for specialist in specialists
        }
        self.max_concurrency = max_concurrency
        """How many specialists of one stage may be in flight at once.

        Held as a number rather than as a ``Semaphore`` because the workflow
        drives each stage through its own ``asyncio.run``, and a semaphore binds
        itself to the first loop that has to make a waiter on it. That binding is
        lazy: an uncontended one never makes a waiter and never notices, so a bus
        that had only ever dispatched four-at-a-time worked for the life of the
        process. Discovery now fans out to ten, the fifth caller waits, and the
        stage after that one died on "bound to a different event loop".
        """

    @property
    def agent_cards(self) -> tuple[AgentCard, ...]:
        return tuple(service.card for service in self.services.values())

    async def dispatch_stage(
        self,
        session: Session,
        specialists: tuple[Specialist, ...],
        *,
        feedback: str = "",
        revision: int = 1,
    ) -> list[TaskResult]:
        # One per dispatch, so it belongs to the loop that is about to use it.
        # The bound is on a single stage's fan-out and the bus runs one stage at
        # a time, so nothing is lost by scoping it to the call.
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def dispatch(specialist: Specialist) -> TaskResult:
            async with semaphore:
                key = (
                    f"{session.id}:{specialist.stage}:{specialist.role}:"
                    f"{revision}:{feedback}"
                )
                existing = next(
                    (
                        task
                        for task in session.tasks
                        if task.idempotency_key == key
                        and task.state == TaskState.COMPLETED
                    ),
                    None,
                )
                if existing and existing.output_artifact_id:
                    artifact = next(
                        item
                        for item in session.artifacts
                        if item.id == existing.output_artifact_id
                    )
                    return TaskResult(task=existing, artifact=artifact)
                return await self.services[specialist.role].execute(
                    session, feedback=feedback, idempotency_key=key
                )

        return list(await asyncio.gather(*(dispatch(item) for item in specialists)))
