from coscientist.models import ApprovalProfile
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

    evolved = presentations["evolve"]
    assert evolved is not None
    assert evolved["evolution"]

    landscape = presentations["proximity"]
    assert landscape is not None
    assert all(cluster["candidates"] for cluster in landscape["clusters"])

    meta = presentations["meta_review"]
    assert meta is not None
    assert meta["recommendations"]
