"""Seed a session halted on three fatal governance findings, and print its id.

The offline provider never writes a fatal flaw, so the browser's governance
path only ever ran when a live model happened to flag one -- which is to say it
ran on production runs and nowhere else. This puts the same state on disk that a
flagged reflect stage leaves behind, so the card that answers it can be driven
deterministically and for nothing.

Three findings rather than one, because the defect this exists to catch only
appears with more than one: answering the first used to tear the card down and
append a new card holding the other two.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coscientist.ledger import ResearchLedger
from coscientist.models import (
    Artifact,
    ArtifactStatus,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    ReviewSet,
    Session,
)

QUESTION = (
    "Does a protective coating improve rechargeable battery cycle life "
    "compared with an uncoated control?"
)

FLAWED = [
    (
        "cand_2",
        "Anneal the assembled electrode at 400 C",
        "Annealing PVDF-containing electrodes at 400 C decomposes the binder "
        "and releases hydrogen fluoride.",
        "No fume extraction is specified.",
    ),
    (
        "cand_3",
        "Cycle pouch cells to 5.2 V to accelerate the fade signal",
        "Charging beyond 4.6 V vents the carbonate electrolyte and can ignite "
        "the cell inside an unvented chamber.",
        "No pressure relief or containment is described.",
    ),
    (
        "cand_4",
        "Dissolve the recovered coating in hydrofluoric acid for assay",
        "Hydrofluoric acid handling is specified with no calcium gluconate on "
        "hand and no named trained operator.",
        "The disposal route for the spent acid is unstated.",
    ),
]


def _candidate(candidate_id: str, title: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        title=title,
        claim=f"{title}.",
        rationale="Because the mechanism predicts it.",
        mechanism_model="The coating blocks the reaction that drives fade.",
        validation_protocol="Coin cells against an uncoated control.",
        predictions=["Capacity retention improves."],
        falsifier="Retention does not improve.",
    )


def main() -> None:
    state_dir = Path(os.environ["COSCIENTIST_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ResearchLedger(state_dir / "research_workflows.db")

    session = Session(question=QUESTION)
    population = CandidatePopulation(
        candidates=[
            _candidate("cand_1", "A conformal alumina coating passivates the surface"),
            *(_candidate(item[0], item[1]) for item in FLAWED),
        ],
        target_size=4,
    )
    reviews = ReviewSet(
        reviews=[
            CandidateReview(
                id=f"rev_{index}",
                candidate_id=candidate_id,
                criterion="safety_governance",
                reviewer="ethics_safety_governance",
                recommendation="reject",
                fatal_flaws=[flaw],
                objections=[objection],
            )
            for index, (candidate_id, _title, flaw, objection) in enumerate(FLAWED, 1)
        ]
    )
    session.artifacts = [
        Artifact(
            stage="generate",
            agent="generation",
            content="",
            schema_name="CandidatePopulation",
            payload=population.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        ),
        Artifact(
            stage="reflect",
            agent="ethics_safety_governance",
            content="",
            schema_name="ReviewSet",
            payload=reviews.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        ),
    ]
    session.status = "governance_blocked"
    session.current_stage = 3  # reflect, where the safety reviewer runs
    store.save(session)
    print(session.id)


if __name__ == "__main__":
    main()
