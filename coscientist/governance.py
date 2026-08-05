"""The way out of a governance block.

The reflect stage halts a session whenever the safety and governance reviewer
records a fatal flaw. That much is right -- a live run stopped on a candidate
proposing to anneal assembled electrodes at 400 C, above the point where the
PVDF binder decomposes and vents hydrogen fluoride, which is a real hazard to a
real person at a real bench. What was missing is the other half: nothing in the
CLI, the TUI or the service let a human answer the block, so the only way past
it was to hand-edit the session file. A gate with no adjudication path does not
make a system safer; it makes the gate the first thing an operator removes.

Two answers are allowed. Withdrawing drops the hypothesis and rewrites the
population without it, keeping the original as superseded history. Overriding
keeps the hypothesis and accepts the flaw. Both demand a named person and a
written reason, and both are replayed in the dossier beside the flaw they
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    Artifact,
    ArtifactStatus,
    CandidatePopulation,
    CandidateReview,
    GovernanceAdjudication,
    ReviewSet,
    Session,
)


@dataclass(frozen=True)
class GovernanceBlocker:
    """A fatal governance finding and the artifact that carries it."""

    review: CandidateReview
    artifact_id: str

    @property
    def review_id(self) -> str:
        return self.review.id

    @property
    def candidate_id(self) -> str:
        return self.review.candidate_id


def governance_blockers(session: Session) -> list[GovernanceBlocker]:
    """Every fatal governance finding in the session, adjudicated or not."""
    blockers: list[GovernanceBlocker] = []
    for artifact in session.artifacts:
        if artifact.agent != "ethics_safety_governance":
            continue
        if artifact.schema_name != "ReviewSet" or not artifact.payload:
            continue
        for review in ReviewSet.model_validate(artifact.payload).reviews:
            if review.fatal_flaws:
                blockers.append(
                    GovernanceBlocker(review=review, artifact_id=artifact.id)
                )
    return blockers


def adjudicated_review_ids(session: Session) -> set[str]:
    return {item.review_id for item in session.governance_adjudications}


def open_blockers(session: Session) -> list[GovernanceBlocker]:
    """Fatal findings nobody has answered yet."""
    answered = adjudicated_review_ids(session)
    return [
        item for item in governance_blockers(session) if item.review_id not in answered
    ]


def withdrawn_candidate_ids(session: Session) -> set[str]:
    return {
        item.candidate_id
        for item in session.governance_adjudications
        if item.resolution == "withdraw"
    }


def latest_population(session: Session) -> Artifact | None:
    for artifact in reversed(session.artifacts):
        if (
            artifact.schema_name == "CandidatePopulation"
            and artifact.payload
            and artifact.status != ArtifactStatus.SUPERSEDED
        ):
            return artifact
    return None


class WithdrawalRefused(ValueError):
    """Raised when withdrawing would leave nothing to reason about."""


def withdraw_candidate(session: Session, candidate_id: str) -> Artifact | None:
    """Replace the population with one that omits ``candidate_id``.

    A new artifact rather than a mutation: the population a stage was reviewed
    against has to stay readable afterwards, so the original is marked
    superseded and kept. Returns the replacement, or ``None`` when the candidate
    was not in the population to begin with.
    """
    current = latest_population(session)
    if current is None:
        return None
    population = CandidatePopulation.model_validate(current.payload)
    remaining = [item for item in population.candidates if item.id != candidate_id]
    if len(remaining) == len(population.candidates):
        return None
    if not remaining:
        # The contract forbids an empty population, and rightly: a session with
        # no hypotheses left has not been fixed by withdrawal, it has failed.
        raise WithdrawalRefused(
            "Withdrawing the last remaining candidate would leave the session "
            "with nothing to rank. Stop the session instead."
        )
    replacement = population.model_copy(deep=True)
    replacement.candidates = remaining
    current.status = ArtifactStatus.SUPERSEDED
    revised = Artifact(
        stage=current.stage,
        agent=current.agent,
        content=current.content,
        artifact_type=current.artifact_type,
        schema_name="CandidatePopulation",
        parent_id=current.id,
        version=current.version + 1,
        input_artifact_ids=list(current.input_artifact_ids),
        producer_model=current.producer_model,
        prompt_version=current.prompt_version,
        payload_source=current.payload_source,
        status=ArtifactStatus.ACCEPTED,
        payload=replacement.model_dump(mode="json"),
    )
    session.artifacts.append(revised)
    return revised


def record_adjudication(
    session: Session,
    blocker: GovernanceBlocker,
    *,
    resolution: str,
    adjudicator: str,
    justification: str,
) -> GovernanceAdjudication:
    adjudication = GovernanceAdjudication(
        review_id=blocker.review_id,
        candidate_id=blocker.candidate_id,
        resolution=resolution,
        adjudicator=adjudicator,
        justification=justification,
        fatal_flaws=list(blocker.review.fatal_flaws),
    )
    session.governance_adjudications.append(adjudication)
    return adjudication
