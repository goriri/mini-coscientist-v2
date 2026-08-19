"""Seed one run that is mid-stage with nothing to decide, and print its id.

This is the state a link to a run in progress arrives in: the evidence stage is
running on a worker somewhere, so there is no draft on the table, no dossier,
and nothing for the researcher to do but watch. It is also the state the page
had no way to reach on its own -- by the time a locally driven run has been
polled twice the deterministic provider has already produced a draft, and a
draft is what used to be the only thing that put the transcript on screen. So
it is written to the ledger rather than played out.

The operation is left without a lease on purpose. ``requeue_expired_operation``
only touches a running operation whose lease has run out, so this one is never
recovered, never advances, and holds the run still for as long as the browser
is looking at it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coscientist.ledger import ResearchLedger
from coscientist.models import Session
from coscientist.orchestration import WORKFLOW_STAGES

QUESTION = "Which host factors govern relapse after antiviral therapy?"
DETAIL = "Deep Research has finished 4 of 7 searches; 0 sources so far."


def main() -> None:
    state_dir = Path(os.environ["COSCIENTIST_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ResearchLedger(state_dir / "research_workflows.db")

    session = Session(question=QUESTION)
    session.status = "active"
    session.current_stage = WORKFLOW_STAGES.index("evidence")
    store.save(session)
    store.set_operation(session.id, "running", DETAIL, "evidence")

    print(json.dumps({"id": session.id, "question": QUESTION, "detail": DETAIL}))


if __name__ == "__main__":
    main()
