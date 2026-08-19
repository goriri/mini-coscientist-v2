"""Seed four finished sessions with staggered clocks, and print them newest first.

The history panel and the search overlay both order by when the research last
moved, and neither ordering can be checked against a server whose sessions were
all written in the same second. So the four are written out of order and then
backdated on disk: the run created second is the newest, the run created third
is the oldest, and any implementation that shows them in insertion order, in
identifier order, or in the order they were last opened disagrees with this
list.

They are seeded finished rather than active on purpose. An unfinished run is one
the landing page takes over to show, and one the browser polls; both would move
a clock this test is holding still.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coscientist.ledger import ResearchLedger
from coscientist.models import Session

# Question, then the day the run last moved. Written in this order, which is
# neither the order they are shown in nor the reverse of it.
SEEDED = [
    (
        "Which antibiotic stewardship interventions reduce resistance in "
        "intensive care?",
        "2026-06-02T09:15:00+00:00",
    ),
    (
        "How does ocean warming change coral reef recovery after bleaching?",
        "2026-08-05T18:40:00+00:00",
    ),
    (
        "What limits the operating lifetime of perovskite photovoltaic cells?",
        "2026-02-14T07:05:00+00:00",
    ),
    (
        "Does a protective coating improve rechargeable battery cycle life?",
        "2026-07-21T11:30:00+00:00",
    ),
]

# Where the report stage sits in the nine-stage workflow, which is where a run
# that has been through everything and is waiting to be printed stops.
REPORT_STAGE = 8


def main() -> None:
    state_dir = Path(os.environ["COSCIENTIST_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    database = state_dir / "research_workflows.db"
    store = ResearchLedger(database)

    seeded = []
    for question, moved_at in SEEDED:
        session = Session(question=question)
        session.status = "ready_for_report"
        session.current_stage = REPORT_STAGE
        store.save(session)
        _backdate(database, session.id, moved_at)
        seeded.append({"id": session.id, "question": question, "updatedAt": moved_at})

    # Newest first, which is what the page has to agree with.
    seeded.sort(key=lambda item: item["updatedAt"], reverse=True)
    print(json.dumps(seeded))


def _backdate(database: Path, session_id: str, moved_at: str) -> None:
    """Move one session's clock, in both places the listing reads it from.

    ``save`` stamps ``utc_now()`` over whatever the session carried, and the
    listing takes ``created_at`` out of the stored payload and ``updated_at``
    out of the column beside it. Setting one and not the other leaves a run
    that sorts by today and displays as February.
    """
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT payload FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        payload = json.loads(row[0])
        payload["created_at"] = moved_at
        payload["updated_at"] = moved_at
        connection.execute(
            "UPDATE sessions SET payload = ?, updated_at = ? WHERE id = ?",
            (json.dumps(payload), moved_at, session_id),
        )


if __name__ == "__main__":
    main()
