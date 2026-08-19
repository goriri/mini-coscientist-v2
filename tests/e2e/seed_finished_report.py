"""Seed a session that has finished and printed, and print its id.

The export row only exists under a finished dossier, and finishing one for real
means driving nine stages. Nothing about the row depends on what is in the
report, so this puts a run past the last stage and lets the renderer produce
whatever a run with no artifacts produces.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coscientist.ledger import ResearchLedger
from coscientist.models import Session
from coscientist.orchestration import WORKFLOW_STAGES

QUESTION = "Does a protective coating improve rechargeable battery cycle life?"


def main() -> None:
    state_dir = Path(os.environ["COSCIENTIST_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ResearchLedger(state_dir / "research_workflows.db")

    session = Session(question=QUESTION)
    session.status = "ready_for_report"
    # Past the last stage, which is what the workflow reads as finished.
    session.current_stage = len(WORKFLOW_STAGES)
    store.save(session)
    print(session.id)


if __name__ == "__main__":
    main()
