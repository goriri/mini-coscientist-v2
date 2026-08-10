"""Starting a run on the corpus an earlier run already searched for.

Discovery is the long half of a run: eight Deep Research passes against one
question, an hour of waiting and a bill to match. Asking the same question a
second time buys all of it again to arrive at the same corpus. A fork skips it
and starts at generation -- and then has to say so, because everything the
Knowledge Base reports about that corpus is another run's work.
"""

import pytest
from fastapi import BackgroundTasks

from app import research_api
from app.research_api import CreateResearchSession, create_research_session
from coscientist.ledger import ResearchLedger
from coscientist.models import ApprovalProfile
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a protective coating improve battery cycle life?"


def _searched(**kwargs) -> CoScientistWorkflow:
    """A run that has cleared scope and evidence and is waiting at generation."""
    flow = CoScientistWorkflow(QUESTION, workflow_version=2, **kwargs)
    flow.accept(flow.preview())
    flow.accept_exploratory_evidence()
    flow.accept(flow.preview())
    assert flow.stage == "generate"
    return flow


def _fresh(question: str = QUESTION, **kwargs) -> CoScientistWorkflow:
    return CoScientistWorkflow(question, workflow_version=2, **kwargs)


def test_a_fork_starts_at_generation_on_the_corpus_the_first_run_built():
    source = _searched()
    fork = _fresh()

    fork.seed_evidence_from(source.session)

    assert fork.stage == "generate"
    assert fork.session.seeded_evidence_from == source.session.id
    # The corpus itself, not a note saying there was one: the generators read the
    # accepted EvidencePacket, and a fork that carried the summary and left the
    # payload behind would reason over nothing and report a full evidence base.
    carried = fork.session.artifact("evidence")
    assert carried is not None
    assert [item.id for item in fork.session.artifacts] == [
        item.id
        for item in source.session.artifacts
        if item.stage in ("scope", "evidence")
    ]


def test_a_fork_records_no_approval_nobody_gave():
    """Nobody in this run read that evidence, and the audit trail is not the place
    to invent a person who did. What it does hold is where the corpus came from."""
    source = _searched()
    fork = _fresh()

    fork.seed_evidence_from(source.session)

    assert fork.session.decisions == []
    seeded = [
        event for event in fork.session.events if event.event_type == "evidence_seeded"
    ]
    assert len(seeded) == 1
    assert seeded[0].payload["source_session_id"] == source.session.id


def test_a_fork_does_not_promise_an_evidence_gate_it_starts_past():
    """The gate stands in the evidence stage, and a fork starts at generation. Asked
    for and recorded, it put a stop in the session the run was never going to make --
    and the launcher, reading the session back, told the person to expect it."""
    source = _searched()
    fork = _fresh(evidence_review=True)

    fork.seed_evidence_from(source.session)

    assert fork.session.evidence_review is False


def test_a_corpus_is_not_carried_to_a_question_it_was_not_searched_against():
    """Coverage is scored per research direction and per facet, both keyed on the
    question. Carried elsewhere, the fork would publish a coverage figure for gaps
    nobody ever searched for."""
    source = _searched()
    fork = _fresh("Can a solid electrolyte improve battery cycle life?")

    with pytest.raises(ValueError, match="different question"):
        fork.seed_evidence_from(source.session)


def test_a_run_that_never_cleared_its_evidence_cannot_be_forked():
    unfinished = CoScientistWorkflow(QUESTION, workflow_version=2)
    unfinished.accept(unfinished.preview())
    fork = _fresh()

    with pytest.raises(ValueError, match="no accepted evidence base"):
        fork.seed_evidence_from(unfinished.session)


def test_a_run_that_has_already_started_cannot_be_seeded():
    """Seeding replaces the scope and evidence half of the record wholesale, so a
    run with work of its own behind it would lose it without being told."""
    source = _searched()
    started = _fresh()
    started.accept(started.preview())

    with pytest.raises(ValueError, match="has not started"):
        started.seed_evidence_from(source.session)


def test_a_fork_cannot_be_its_own_source():
    """There is no guard spelled "not from itself", and adding one would be dead
    code: a run holding a corpus to fork has by definition started, and one that
    has not started holds nothing to fork. Both ways round are already refused --
    what matters is that neither silently succeeds and reseeds a run onto itself."""
    source = _searched()
    with pytest.raises(ValueError, match="has not started"):
        source.seed_evidence_from(source.session)

    fresh = _fresh()
    with pytest.raises(ValueError, match="no accepted evidence base"):
        fresh.seed_evidence_from(fresh.session)


# ---------------------------------------------------------------------------
# The launcher, which is where the id gets pasted in
# ---------------------------------------------------------------------------


@pytest.fixture()
def api(tmp_path, monkeypatch):
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    return store


def _created(**body) -> dict:
    return create_research_session(
        CreateResearchSession(question=QUESTION, **body), BackgroundTasks()
    )


def test_the_launcher_forks_the_run_whose_id_was_pasted_in(api):
    source = _searched(approval_profile=ApprovalProfile.MILESTONE, ledger=api)

    snapshot = _created(seed_evidence_from=source.session.id)

    assert snapshot["stage"] == "generate"


def test_an_id_nobody_holds_is_a_typo_in_the_field_and_says_so(api):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        _created(seed_evidence_from="session_nothing_here")

    assert raised.value.status_code == 404


def test_a_run_that_cannot_be_forked_is_reported_apart_from_one_that_is_missing(api):
    """Reported as one status they were indistinguishable in the launcher: a typo
    and a badly chosen run need different things done about them."""
    from fastapi import HTTPException

    unfinished = CoScientistWorkflow(QUESTION, workflow_version=2, ledger=api)
    unfinished.accept(unfinished.preview())

    with pytest.raises(HTTPException) as raised:
        _created(seed_evidence_from=unfinished.session.id)

    assert raised.value.status_code == 400
    assert "no accepted evidence base" in raised.value.detail
