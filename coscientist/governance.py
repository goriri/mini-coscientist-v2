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

from collections.abc import Sequence
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


REHEARSAL_ADJUDICATOR = "rehearsal (nobody)"
"""Deliberately not a name, because no person read the finding.

Every other adjudicator string in the system is somebody who signed for a
hazard. This one has to be impossible to mistake for one of those at a glance,
in a table, a year later.
"""

REHEARSAL_JUSTIFICATION = (
    "NOT REVIEWED. This run is a rehearsal of the pipeline, not a research "
    "proposal: it was launched to exercise the workflow end to end and nothing "
    "in it is intended for execution. No person read this finding and nothing "
    "here answers it. The flaw stands exactly as the reviewer wrote it, and any "
    "hypothesis carrying one must go through this gate for real -- with a named "
    "adjudicator -- before it is proposed to anybody."
)


def rehearsal_adjudications(
    blockers: Sequence[GovernanceBlocker],
) -> list[GovernanceAdjudication]:
    """Answer a rehearsal's own gate, in writing, without pretending anyone did.

    Called only where the reflect stage would otherwise stop the run. Each finding
    it is given is recorded as an override so the run can continue, and each one
    carries the same sentence saying no one read it -- which is the honest thing
    for the dossier to print, and the useful one: an override signed by a person
    and an override signed by nobody are different claims about the same flaw,
    and only one of them was checked.

    Takes the blockers rather than the session, so it waives exactly what the
    caller is stopped on. The reflect gate is scoped to the findings that came
    out of the review set it is admitting; reading ``open_blockers`` again here
    would sign off a finding from some other artifact that nothing had yet asked
    about.

    Overrides rather than withdrawals on purpose. Withdrawing would drop the
    hypothesis and rewrite the population around it, so the rehearsal would stop
    exercising the stages it exists to exercise -- and would quietly produce a
    smaller report than the real run it stands in for.
    """
    return [
        GovernanceAdjudication(
            review_id=blocker.review_id,
            candidate_id=blocker.candidate_id,
            resolution="override",
            adjudicator=REHEARSAL_ADJUDICATOR,
            justification=REHEARSAL_JUSTIFICATION,
            fatal_flaws=list(blocker.review.fatal_flaws),
        )
        for blocker in blockers
    ]


def blockers_for_draft(
    session: Session, draft: Artifact | None
) -> list[GovernanceBlocker]:
    """The fatal findings a given stage draft is the gate for.

    The reflect gate stops on the findings that came out of the review set it is
    admitting, which is ``blocker.artifact_id in draft.input_artifact_ids`` --
    a finding belonging to some other review set is not what this gate is
    asking about. Every reader of the gate has to scope it the same way, and
    the web card did not: it drew every fatal finding in the session.

    A reflect revision is enough to make the two disagree. The re-review is a
    new review set, and any finding left unanswered in the one it replaced
    stays in the session forever. A live run reached the tournament with eight
    of those behind it and the card said "8 safety findings unanswered" over
    the ranking gate, greyed out the accept button for a stage that had nothing
    to do with them, and refused every override pressed on them -- adjudication
    applies only to a blocked session, and the session was not blocked. Under
    the human profile that is a run with no way forward on screen at all.
    """
    if draft is None:
        return []
    inputs = set(draft.input_artifact_ids)
    return [item for item in governance_blockers(session) if item.artifact_id in inputs]


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
