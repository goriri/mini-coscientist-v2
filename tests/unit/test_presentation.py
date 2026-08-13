from coscientist.models import (
    ApprovalProfile,
    Artifact,
    Candidate,
    CandidatePopulation,
    DeepResearchRun,
    DiscoveryManifest,
    DossierManifest,
    EvolutionCycle,
    EvolutionRecord,
    Session,
    SourceLead,
)
from coscientist.narrative import idea_title
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
    # Every card is headed by the idea it is a verdict on. "Candidate 5", or the
    # raw id of an idea evolution had superseded, is not a verdict anyone can
    # check -- and the panel carries nothing else to identify it by.
    assert all(item["title"] for item in meta["recommendations"])
    assert not any(
        item["title"].startswith(("cand_", "Candidate "))
        for item in meta["recommendations"]
    )
    # And the last rows of the last panel are section names, not the keys the
    # dossier writer files them under.
    sections = next(
        item for item in meta["details"] if item["label"] == "Dossier sections"
    )
    assert sections["value"] and all(
        isinstance(title, str) for title in sections["value"]
    )


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


def test_a_dropped_idea_is_named_in_the_panel_that_says_it_was_dropped():
    """Withdrawal rewrites the population without the idea it dropped.

    Names were read off the live population only, so the meta-review card put
    "Excluded" over ``cand_two`` -- the identifier the panel exists to spare the
    reader -- for every hypothesis a safety finding had removed.
    """
    session = Session(question="Can a protective coating help?")

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

    kept = _candidate("cand_one", "A conformal alumina coating passivates the surface")
    dropped = _candidate("cand_two", "Anneal the assembled electrode at 400 C")
    for population in (
        CandidatePopulation(candidates=[kept, dropped], target_size=2),
        CandidatePopulation(candidates=[kept], target_size=2),
    ):
        session.artifacts.append(
            Artifact(
                stage="generate",
                agent="generation",
                content="",
                schema_name="CandidatePopulation",
                payload=population.model_dump(mode="json"),
            )
        )
    session.artifacts.append(
        Artifact(
            stage="meta_review",
            agent="meta_review",
            content="",
            schema_name="DossierManifest",
            payload=DossierManifest(
                title="Coatings for cycle life",
                sections=[],
                recommendation_candidate_ids=["cand_one"],
                unresolved_fatal_flaw_candidate_ids=["cand_two"],
            ).model_dump(mode="json"),
        )
    )

    presentation = build_stage_presentation(session, "meta_review")
    assert presentation is not None
    excluded = presentation["recommendations"][1]
    assert excluded["candidate_id"] == "cand_two"
    # The name the dossier gives it, so the two do not disagree.
    assert excluded["title"] == idea_title(dropped)
    assert excluded["title"] == "Anneal the Assembled Electrode at 400 C"
    # And the one still in the field is headed by its name, not its position in
    # it: "Candidate 1" is a verdict a reader cannot check.
    assert presentation["recommendations"][0]["label"] == "Candidate 1"
    assert presentation["recommendations"][0]["title"] == idea_title(kept)


def test_an_evolved_idea_is_named_in_the_panel_that_recommends_it():
    """Evolution writes its rewrites under new ids and no new population.

    So the ideas the meta-review actually reasons about -- the evolved ones --
    were the ones it could not name. A live run printed four of them, all four
    as identifiers.
    """
    session = Session(question="Can a protective coating help?")
    parent = Candidate(
        id="cand_one",
        title="A conformal alumina coating passivates the surface",
        claim="A conformal alumina coating passivates the surface.",
        rationale="Because the mechanism predicts it.",
        mechanism_model="The coating blocks the reaction that drives fade.",
        validation_protocol="Coin cells against an uncoated control.",
        predictions=["Capacity retention improves."],
        falsifier="Retention does not improve.",
    )
    evolved = parent.model_copy(
        update={
            "id": "cand_one_03",
            "title": "An atomic-layer alumina coating passivates the surface",
        }
    )
    session.artifacts.append(
        Artifact(
            stage="generate",
            agent="generation",
            content="",
            schema_name="CandidatePopulation",
            payload=CandidatePopulation(candidates=[parent], target_size=1).model_dump(
                mode="json"
            ),
        )
    )
    session.artifacts.append(
        Artifact(
            stage="evolve",
            agent="evolution",
            content="",
            schema_name="EvolutionCycle",
            payload=EvolutionCycle(
                records=[
                    EvolutionRecord(
                        parent_ids=["cand_one"],
                        candidate=evolved,
                        changes=["Deposited by atomic layer deposition instead."],
                        new_prediction="Retention improves further.",
                        round_number=3,
                    )
                ]
            ).model_dump(mode="json"),
        )
    )
    session.artifacts.append(
        Artifact(
            stage="meta_review",
            agent="meta_review",
            content="",
            schema_name="DossierManifest",
            payload=DossierManifest(
                title="Coatings for cycle life",
                sections=[],
                recommendation_candidate_ids=["cand_one_03"],
            ).model_dump(mode="json"),
        )
    )

    presentation = build_stage_presentation(session, "meta_review")
    assert presentation is not None
    recommended = presentation["recommendations"][0]
    assert recommended["candidate_id"] == "cand_one_03"
    assert recommended["title"] == idea_title(evolved)
    assert recommended["label"] == idea_title(evolved)
