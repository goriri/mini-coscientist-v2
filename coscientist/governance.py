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

import re
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


_FLAW_STOPWORDS = frozenset(
    """this that these those with without from into onto over under
    which where when what have been being will would could should must
    them they their there here than then also more most some such very
    lacks lack lacking fails fail failed missing absence absent does
    require requires required need needs needed provide provides state
    states stated specify specifies specified explicit explicitly
    proposed proposal propose candidate hypothesis study review finding""".split()
)
"""Grammar and reviewer's phrasing, so a paraphrase is not read as a new hazard."""


def _flaw_topic(text: str) -> set[str]:
    """What a fatal flaw is about, with the wording taken off.

    Its own rather than the report's: ``narrative`` has a more careful version of
    this and imports from here, so taking that one would close the loop. What is
    matched here is a hazard rather than an objection, and the two want different
    stopwords anyway.
    """
    words = set()
    for token in re.findall(r"[a-z]+", text.lower()):
        if len(token) < 4 or token in _FLAW_STOPWORDS:
            continue
        for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                token = token[: -len(suffix)]
                break
        words.add(token)
    return words


def _same_flaw(one: str, other: str) -> bool:
    """Whether two fatal flaws are the same hazard written twice.

    Two content words in common is the floor, and the shorter of the two has to
    be half covered, so "vents hydrogen fluoride above the binder's decomposition
    point" is not merged with "no statistical power calculation" over a word they
    happen to share.
    """
    first, second = _flaw_topic(one), _flaw_topic(other)
    if not first or not second:
        return False
    overlap = len(first & second)
    return overlap >= 2 and overlap / min(len(first), len(second)) >= 0.5


def is_answered(session: Session, blocker: GovernanceBlocker) -> bool:
    """Whether a named person has already signed for this hazard.

    By review id, and failing that by the hazard itself. A reflect revision
    re-reviews the same population and the reviewer writes its findings fresh, so
    the same flaw on the same hypothesis comes back under a new id -- one live run
    raised ``rev_cand_analogy_sam_ald_safety``, took an override for it, re-ran
    reflect, and blocked again on ``rev_safety_cand_analogy_sam_ald``: the same
    sentence about the same candidate, permuted. Nothing about answering it moved
    the run forward, and it would have gone round for as long as somebody kept
    pressing override.

    Every flaw in the finding has to be covered, not just one. A re-review that
    turns up a hazard nobody signed for is a new finding and blocks again, which
    is the half of this that keeps the gate a gate.
    """
    if blocker.review_id in adjudicated_review_ids(session):
        return True
    signed = [
        flaw
        for item in session.governance_adjudications
        if item.candidate_id == blocker.candidate_id
        for flaw in item.fatal_flaws
    ]
    if not signed:
        return False
    return all(
        any(_same_flaw(flaw, prior) for prior in signed)
        for flaw in blocker.review.fatal_flaws
    )


def open_blockers(session: Session) -> list[GovernanceBlocker]:
    """Fatal findings nobody has answered yet."""
    return [
        item for item in governance_blockers(session) if not is_answered(session, item)
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
