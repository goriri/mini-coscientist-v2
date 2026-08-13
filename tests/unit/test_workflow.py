import json
from pathlib import Path

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.models import ApprovalMode, ArtifactStatus
from coscientist.orchestration import CoScientistWorkflow


def test_human_mode_requires_current_explicit_acceptance(tmp_path: Path):
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        DeterministicProvider(),
        approval_mode=ApprovalMode.HUMAN,
        workflow_version=1,
    )
    first = flow.preview()
    assert flow.stage == "scope"
    assert first.stage == "scope"
    assert flow.session.current_stage == 0

    with pytest.raises(ValueError, match="Automatic decisions are disabled"):
        flow.accept(first, automatic=True)

    revised = flow.revise("Require a 500-cycle endpoint")
    assert revised.feedback == "Require a 500-cycle endpoint"
    assert revised.parent_id == first.id
    assert flow.session.current_stage == 0
    assert first.status == ArtifactStatus.SUPERSEDED

    with pytest.raises(ValueError, match="latest"):
        flow.accept(first)
    flow.accept(revised, actor="test_researcher")
    assert flow.stage == "generate"
    assert flow.session.decisions[-1].automatic is False
    assert flow.session.decisions[-1].actor == "test_researcher"

    while not flow.done:
        flow.accept(flow.preview(), actor="test_researcher")
    report = flow.render_report()
    # The report has to end on a recommendation, state which approval regime
    # produced it, and carry the integrity caveat that nothing in it is a finding.
    # The caveat is in the warnings chapter now; every gate here was answered by a
    # person, so the auto-approval warning is correctly absent from this one.
    assert "#### 9. Recommendations and Next Steps" in report
    assert "Approval profile:" in report
    assert "# Warnings and Limitations" in report
    assert "Nothing in this document is a finding" in report
    assert "Auto approval is a workflow convenience" not in report
    assert len(
        [
            artifact
            for artifact in flow.session.artifacts
            if artifact.artifact_type == "stage_bundle"
            and artifact.status == ArtifactStatus.ACCEPTED
        ]
    ) == len(flow.workflow_stages)

    path = tmp_path / "session.json"
    flow.save(path)
    loaded = CoScientistWorkflow.load(path)
    assert loaded.session.question == flow.session.question
    assert loaded.done
    assert json.loads(path.read_text())["status"] == "ready_for_report"


def test_auto_mode_completes_and_audits_automatic_decisions():
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        DeterministicProvider(),
        approval_mode=ApprovalMode.AUTO,
        workflow_version=1,
    )
    flow.run_auto()

    assert flow.done
    assert flow.session.status == "ready_for_report"
    accepts = [
        decision for decision in flow.session.decisions if decision.action == "accept"
    ]
    assert len(accepts) == len(flow.workflow_stages)
    assert all(decision.automatic for decision in accepts)
    assert all(decision.actor == "auto_approval_policy" for decision in accepts)
    assert "Auto approval is a workflow convenience" in flow.render_report()
    governance = next(
        item
        for item in flow.session.artifacts
        if item.agent == "ethics_safety_governance"
    )
    assert "HUMAN REVIEW REQUIRED" in governance.content


def test_only_discovery_card_has_google_search():
    flow = CoScientistWorkflow("A research question")
    cards_with_search = [
        card.name for card in flow.agent_cards if "google_search" in card.tools
    ]
    assert cards_with_search == ["evidence_discovery"]
    verifier = next(
        card for card in flow.agent_cards if card.name == "source_verification"
    )
    # The verifier retrieves through the in-repo fetcher, which records the HTTP
    # status and the registry lookup. ``load_web_page`` returns page text with no
    # provenance, and a verifier that only sees text cannot tell a paywall notice
    # from a paper.
    assert verifier.tools == ["fetch_source_document"]
    assert len(flow.agent_cards) == 18


def test_a_drafted_stage_records_how_long_its_specialists_took():
    """Timing a run meant subtracting neighbouring rows out of ``audit_events``.

    That measures the gap between two stages rather than the work inside one, so
    a run left at an approval gate overnight was indistinguishable from a stage
    that ran all night, and "which stage is slow" could not be answered from the
    trail the run already writes.
    """
    flow = CoScientistWorkflow("Can a coating improve cycle life?")
    flow.preview()

    drafted = next(
        event for event in flow.session.events if event.event_type == "stage_drafted"
    )
    assert drafted.payload["seconds"] >= 0


def test_stopped_session_cannot_advance():
    flow = CoScientistWorkflow("A research question")
    flow.preview()
    flow.stop()
    assert flow.session.status == "stopped_by_researcher"
    with pytest.raises(ValueError, match="cannot advance"):
        flow.preview("try again")


def test_researcher_can_directly_edit_a_draft():
    flow = CoScientistWorkflow("Can a coating improve cycle life?")
    original = flow.preview()

    edited = flow.edit_draft(
        original.content + "\n\nResearcher edit: use a matched control.",
        actor="test_researcher",
    )

    assert original.status == ArtifactStatus.SUPERSEDED
    assert edited.parent_id == original.id
    assert edited.version == 2
    assert edited.producer_model == "human-edited"
    assert flow.pending_draft == edited
    assert flow.session.decisions[-1].action == "revise"
    assert flow.session.decisions[-1].actor == "test_researcher"
