from coscientist.models import (
    ApprovalProfile,
    Artifact,
    DeepResearchRun,
    DiscoveryManifest,
    Session,
    SourceLead,
)
from coscientist.orchestration import CoScientistWorkflow
from coscientist.presentation import build_stage_presentation


def test_every_stage_has_a_structured_human_presentation():
    workflow = CoScientistWorkflow(
        "Can a protective coating improve battery cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    workflow.run_auto()

    presentations = {
        stage: build_stage_presentation(workflow.session, stage)
        for stage in workflow.workflow_stages
    }
    assert all(presentations.values())
    assert all(
        presentation["schema_version"] == "1"
        for presentation in presentations.values()
        if presentation
    )

    generated = presentations["generate"]
    assert generated is not None
    assert len(generated["candidates"]) == 8
    assert all(candidate["claim"] for candidate in generated["candidates"])
    assert all(candidate["falsifier"] for candidate in generated["candidates"])

    reflected = presentations["reflect"]
    assert reflected is not None
    assert all(candidate["reviews"] for candidate in reflected["candidates"])

    ranked = presentations["rank"]
    assert ranked is not None
    assert len(ranked["ranking"]) == 8
    assert len([item for item in ranked["ranking"] if item["shortlisted"]]) == 4
    assert all(item["claim"] for item in ranked["ranking"])
    assert all(candidate["rationale"] for candidate in ranked["candidates"])
    assert ranked["comparison_rounds"]
    # The panel showed four numbers and the rounds behind a fold, which says a
    # ranking happened without saying what it decided. The author travels with
    # the text: this run was decided by arithmetic, and the briefing must not be
    # read as a judge's verdict on the field.
    assert "Final standings:" in ranked["briefing"]
    assert ranked["briefing_author"] == "computed"

    evolved = presentations["evolve"]
    assert evolved is not None
    assert evolved["evolution"]

    landscape = presentations["proximity"]
    assert landscape is not None
    assert all(cluster["candidates"] for cluster in landscape["clusters"])

    meta = presentations["meta_review"]
    assert meta is not None
    assert meta["recommendations"]


def _evidence_view(manifest: DiscoveryManifest) -> dict:
    session = Session(question=manifest.question)
    session.artifacts.append(
        Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            artifact_type="specialist_output",
            content="discovery",
            schema_name="DiscoveryManifest",
            payload=manifest.model_dump(mode="json"),
        )
    )
    view = build_stage_presentation(session, "evidence")
    assert view is not None
    return view


def _labelled(view: dict, label: str):
    return next(
        (item["value"] for item in view["details"] if item["label"] == label), None
    )


def test_the_evidence_view_names_the_provider_that_found_the_leads():
    """Calling a search hit "Deep Research" buys it credibility it did not earn."""
    view = _evidence_view(
        DiscoveryManifest(
            question="Can a coating extend cycle life?",
            source_leads=[
                SourceLead(
                    canonical_url="https://doi.org/10.1000/x", provider="google_search"
                )
            ],
            stored_interaction_notice=False,
        )
    )

    assert "google_search" in view["summary"]
    assert "Deep Research" not in view["summary"]
    assert _labelled(view, "Discovery provider") == ["google_search"]
    assert _labelled(view, "Stored interaction notice") is None


def test_the_evidence_view_keeps_the_notice_when_something_was_stored():
    view = _evidence_view(
        DiscoveryManifest(
            question="Can a coating extend cycle life?",
            source_leads=[SourceLead(canonical_url="https://doi.org/10.1000/x")],
        )
    )

    assert "deep_research" in view["summary"]
    assert _labelled(view, "Stored interaction notice")


def test_the_evidence_view_does_not_pretend_an_empty_pass_found_a_landscape():
    view = _evidence_view(DiscoveryManifest(question="Can a coating help?"))

    assert "nothing was discovered" in view["summary"]
    assert _labelled(view, "Discovery provider") == ["none -- no pass was attempted"]


def test_a_wave_that_returned_no_lead_is_not_reported_as_no_provider():
    """The provider is read off the leads, so an empty wave names none of them.

    Beside a panel reporting seven attempted passes, a bare "none" reads as a
    deployment with no discovery configured. That is a broken install, not a search
    that came back empty, and on a live run it sent the first diagnosis the wrong way.
    """
    view = _evidence_view(
        DiscoveryManifest(
            question="Can a coating help?",
            runs=[
                DeepResearchRun(
                    pass_number=number,
                    interaction_id=f"interaction-{number}",
                    status="completed",
                )
                for number in (1, 2)
            ],
        )
    )

    assert _labelled(view, "Discovery provider") == [
        "none named -- no pass returned a lead"
    ]
