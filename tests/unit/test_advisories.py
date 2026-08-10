"""The warnings chapter, and what the body keeps once the warnings leave it.

Each of these paragraphs used to be printed beside the material it qualifies. That
was the right call one warning at a time and the wrong one in aggregate: a live
report opened with three of them before its first sentence about the science, and
the per-idea copy of the fallback notice was printed eight times. They are collected
into one appendix chapter now, so what these tests hold is the two halves of that
bargain -- nothing run-level is left loose in the body, and nothing idea-level was
swept into the appendix with it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from coscientist.advisories import advisory_pointer, run_advisories
from coscientist.dossier import compile_dossier
from coscientist.models import (
    Artifact,
    ArtifactStatus,
    DecisionAction,
    HumanDecision,
    Session,
)
from coscientist.narrative import (
    BlockerNote,
    IdeaBrief,
    ResearchRecord,
    build_idea_briefs,
    load_record,
)

CHAPTER = "\n# Warnings and Limitations"


def _halves(session: Session) -> tuple[str, str]:
    report = compile_dossier(session)
    return report.split(CHAPTER)[0], report[report.index(CHAPTER) :]


def _auto_approved(session: Session) -> Session:
    session.decisions.append(
        HumanDecision(
            action=DecisionAction.ACCEPT,
            stage="generate",
            actor="auto_policy",
            automatic=True,
            session_version=session.version,
        )
    )
    return session


def test_the_chapter_is_present_on_a_run_with_nothing_wrong_with_it():
    """The standing limits are true of every run, so the chapter is never absent.

    A chapter that appears only on troubled runs teaches a reader that its absence
    means the report was checked, which is the one thing it does not mean.
    """
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    advisories = run_advisories(record)

    assert [item.title for item in advisories] == ["What this report is and is not"]
    assert not any(item.blocking for item in advisories)
    pointer = advisory_pointer(advisories)
    assert pointer.startswith("One limitation applies")
    assert "it does not block the work proposed below" in pointer


def test_the_body_pointer_names_the_blocking_warnings_rather_than_counting_them(
    rich_session: Session,
):
    """A bare count gives a reader no reason to turn to the appendix.

    Two reports can carry the same number of limitations and differ entirely in
    whether either of them says to stop, so the count that leads is the blocking one
    and the blockers are named where the reader is.
    """
    rich_session.exploratory_evidence_accepted = True
    body, appendix = _halves(rich_session)

    assert "a waived evidence gate" in body
    assert "ideas citing evidence that does not exist or was retracted" in body
    assert "should not proceed on the material they name" in body
    # Named, not reproduced: the paragraph itself is in one place only.
    assert "Waiving the gate is a distinct act" not in body
    assert "Waiving the gate is a distinct act" in appendix


@pytest.mark.parametrize(
    ("mutate", "moved"),
    [
        (
            lambda session: setattr(session, "exploratory_evidence_accepted", True),
            "The evidence gate for this run was waived",
        ),
        (_auto_approved, "was approved automatically under the auto approval"),
        (lambda session: None, "Nothing in this document is a finding"),
        (lambda session: None, "No experiment was performed in this run"),
        (lambda session: None, "a fixed template, not the specialist's own reasoning"),
        (
            lambda session: None,
            "cites evidence that does not exist in this session",
        ),
        # The other half of the same roll-up. Kept as its own case because the two
        # verdicts it covers fail for different reasons and are worded separately.
        (
            lambda session: None,
            "cites evidence that was retracted, namely",
        ),
    ],
)
def test_every_run_level_warning_is_in_the_chapter_and_not_in_the_body(
    rich_session: Session, mutate, moved: str
):
    mutate(rich_session)
    body, appendix = _halves(rich_session)

    assert moved in appendix
    assert moved not in body


def test_a_caveat_about_one_idea_stays_under_that_idea(rich_session: Session):
    """The line the collection must not cross.

    A grounding verdict and a fatal flaw are findings about one hypothesis. Hoisting
    those into a chapter at the end would mean a reader comparing two ideas could not
    see which of them carries one.
    """
    body, appendix = _halves(rich_session)
    briefs = build_idea_briefs(load_record(rich_session))
    alarming = [brief for brief in briefs if brief.support_is_alarming]

    assert alarming, "the fixture no longer exercises a broken grounding verdict"
    for brief in alarming:
        assert brief.title in body
    assert "Evidence support:" in body
    # The run-level roll-up names them too, which is the point of a roll-up; what it
    # must not do is be the only place they are named.
    assert "unsupported rather than evidence-backed" in appendix


def _grounding_briefs(
    rich_session: Session, verdicts: list[str], cause: str
) -> tuple[IdeaBrief, ...]:
    """The run's leading ideas given `verdicts`, discredited ones for `cause`."""
    original = tuple(build_idea_briefs(load_record(rich_session)))
    assert len(original) >= len(verdicts), "the fixture no longer has ideas to mark"
    marked = list(verdicts) + ["grounded"] * (len(original) - len(verdicts))
    return tuple(
        replace(
            brief,
            support=verdict,
            discrediting_statuses=(
                frozenset({cause}) if verdict == "discredited" else frozenset()
            ),
            unresolved_evidence_ids=["claim_001"] if verdict == "unsupported" else [],
        )
        for brief, verdict in zip(original, marked, strict=True)
    )


def _grounding_advisory(
    rich_session: Session, verdicts: list[str], cause: str = "retracted"
) -> str:
    """The broken-grounding paragraph, with the run's leading ideas given `verdicts`."""
    record = load_record(rich_session)
    briefs = _grounding_briefs(rich_session, verdicts, cause)
    bodies = [
        item.body
        for item in run_advisories(record, briefs=briefs)
        if item.title.startswith("Ideas citing evidence")
    ]
    assert len(bodies) == 1, "the broken-grounding advisory did not fire once"
    return bodies[0]


def test_a_retracted_citation_is_not_reported_as_one_that_resolves_to_nothing(
    rich_session: Session,
):
    """Both verdicts were rolled up under the reason that only holds for one.

    A live report told four ideas here that the evidence they cite "does not exist in
    this session" and told each of them under its own heading that the source is a
    named paper whose claim is discredited. A discredited citation does resolve; the
    record it resolves to was retracted or could not be retrieved. Both statements
    were in the same report about the same four ideas, and nothing distinguished them.
    """
    briefs = tuple(build_idea_briefs(load_record(rich_session)))
    absent_titles = [briefs[0].title, briefs[2].title]
    retracted_titles = [briefs[1].title, briefs[3].title]
    body = _grounding_advisory(
        rich_session, ["unsupported", "discredited", "unsupported", "discredited"]
    )

    missing, _, retracted = body.partition("cite evidence that was retracted")
    assert retracted, "the retracted group was never given its own reason"
    for title in absent_titles:
        assert title in missing and title not in retracted
    for title in retracted_titles:
        assert title in retracted and title not in missing
    assert "resolve to no record" in missing
    assert "do resolve to records, and those records have since been retracted" in (
        retracted
    )


def test_a_group_that_was_only_unreachable_is_not_told_its_sources_were_retracted(
    rich_session: Session,
):
    """The support word folds retraction and unretrievability into one, and the
    sentence written for it asserted the first of them over both.

    Every one of four ideas in a live report had cited a claim that merely failed to
    come back -- Evidence integrity called each "the unretrieved claim drawn from"
    its source -- and the blocking warning told the reader "those citations do resolve
    to records, and the records no longer stand".
    """
    body = _grounding_advisory(rich_session, ["discredited"], cause="inaccessible")

    assert "cites evidence that could not be retrieved" in body
    assert (
        "That citation names a document this run could not reach when it went back "
        "to it, so nothing was read there and nothing was confirmed" in body
    )
    assert "retracted" not in body
    assert "no longer stands" not in body


def test_a_run_whose_only_broken_grounding_was_retracted_is_not_told_it_cited_nothing(
    rich_session: Session,
):
    body = _grounding_advisory(rich_session, ["discredited"])

    assert "does not exist in this session" not in body
    assert "resolve to no record" not in body
    assert "discredited rather than evidence-backed" in body


def test_the_heading_over_the_broken_grounding_claims_no_more_than_the_group_holds(
    rich_session: Session,
):
    """The heading asserted the stronger of the two cases the paragraph reports.

    A live report opened on "one says the work should not proceed on the material it
    names: ideas citing evidence that has been retracted", and every one of the four
    ideas it went on to list had cited a claim that could not be retrieved -- no
    retraction anywhere among them. Evidence integrity in the same report called each
    of the four "the unretrieved claim drawn from ...".
    """
    record = load_record(rich_session)

    def _headings(verdicts: list[str], cause: str = "retracted") -> list[str]:
        briefs = _grounding_briefs(rich_session, verdicts, cause)
        return [
            item.title
            for item in run_advisories(record, briefs=briefs)
            if item.title.startswith("Ideas citing evidence")
        ]

    assert _headings(["discredited"]) == ["Ideas citing evidence that was retracted"]
    assert _headings(["discredited"], cause="inaccessible") == [
        "Ideas citing evidence that could not be retrieved"
    ]
    assert _headings(["unsupported"]) == ["Ideas citing evidence that does not exist"]
    assert _headings(["unsupported", "discredited"]) == [
        "Ideas citing evidence that does not exist or was retracted"
    ]


def test_a_single_broken_idea_is_written_about_in_the_singular(rich_session: Session):
    """ "That citation was written by the generator and resolve to no record" -- the
    count switched the noun and left the verb and the possessive behind it."""
    for verdict in ("unsupported", "discredited"):
        body = _grounding_advisory(rich_session, [verdict])

        assert "whatever their rank says" not in body
        assert "whatever its rank says" in body
        assert "citation was written by the generator and resolves" in body or (
            "citation does resolve to a record, and that record has since been "
            "retracted" in body
        )


def test_an_unanswered_governance_block_leads_the_chapter(rich_session: Session):
    """Order is by consequence. Nothing outranks work a reviewer said to stop."""
    rich_session.exploratory_evidence_accepted = True
    record = load_record(rich_session)
    record.open_governance_blocks = [
        BlockerNote(
            candidate_id="candidate_0001",
            title="A Conformal Alumina Coating Suppresses Electrolyte Decomposition",
            fatal_flaws=["The disposal route for the fluorinated coating is unstated"],
        )
    ]
    advisories = run_advisories(
        record, briefs=tuple(build_idea_briefs(load_record(rich_session)))
    )

    assert advisories[0].title == "Unanswered fatal governance findings"
    assert advisories[0].blocking
    assert "blocked, not approved" in advisories[0].body
    assert [item.blocking for item in advisories].count(True) == 3


def test_the_chapter_opens_the_appendix_rather_than_closing_it(rich_session: Session):
    """Provenance is for auditing the report; this is for deciding whether to act on
    it, so a reader who stops after one chapter has read the right one."""
    report = compile_dossier(rich_session)

    assert report.index(CHAPTER) < report.index("\n# Provenance")


def test_a_fallback_is_warned_about_once_however_many_ideas_the_run_produced(
    rich_session: Session,
):
    """It used to be printed under every overview section the stage fed and again
    under every idea, which on the live report was six copies of one paragraph."""
    report = compile_dossier(rich_session)
    notice = "a fixed template, not the specialist's own reasoning"

    assert report.count(notice) == 1


def test_a_stage_that_did_not_fall_back_raises_no_template_warning():
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.provenance = []

    titles = [item.title for item in run_advisories(record)]

    assert "Stages that produced a template rather than reasoning" not in titles


def test_a_report_with_no_accepted_artifacts_still_compiles_its_chapter():
    """The empty run is the one a compiler is most likely to trip over, and it is
    also the run whose reader most needs to be told nothing here is a finding."""
    session = Session(question="Can a coating help?")
    session.artifacts.append(
        Artifact(
            stage="scope",
            agent="scope_agent",
            artifact_type="stage_bundle",
            schema_name="ResearchPlan",
            content="",
            payload={},
            status=ArtifactStatus.DRAFT,
        )
    )

    report = compile_dossier(session)

    assert CHAPTER in report
    assert "Nothing in this document is a finding" in report


def test_the_waived_gate_says_why_its_total_is_not_the_references_total():
    """Two counts of two populations, four chapters apart, opening the same way.

    Section eight counts the documents the literature search reached and the waiver
    counts the sources the evidence stage admitted. On a live run those were eighty
    and seventy-seven, both introduced as "Twenty-five of the", and nothing anywhere
    said they were counts of different things.
    """
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import EvidencePacket, SourceLead, SourceRecord
    from coscientist.narrative import CitationRegistry

    session = Session(question="Does a coating change cycle life?")
    session.exploratory_evidence_accepted = True
    record = ResearchRecord(session=session)
    record.evidence = EvidencePacket(
        question=session.question,
        sources=[
            SourceRecord(url="https://a.example/paper", verification_status="verified"),
            SourceRecord(url="https://b.example/paper"),
            # Recorded without a link, so the reference list cannot number it.
            SourceRecord(url=""),
        ],
    )
    record.citations = CitationRegistry(
        [
            SourceLead(canonical_url="https://a.example/paper", title="A"),
            SourceLead(canonical_url="https://b.example/paper", title="B"),
            SourceLead(canonical_url="https://c.example/paper", title="C"),
            SourceLead(canonical_url="https://d.example/paper", title="D"),
        ]
    )
    body = _waived_gate_advisory(record)[0].body

    assert "Of the three sources admitted, one was retrieved and checked" in body
    assert "that one is marked as verified where it is cited" in body
    assert (
        "That total counts what the evidence stage admitted, which is not the " in body
    )
    assert (
        "four documents the literature search reached that Key Findings counts" in body
    )
    assert "two documents there are leads the search returned that no admitted " in body
    assert "one source admitted here names no document that can be numbered" in body


def test_the_waived_gates_four_counts_are_written_in_one_number_style():
    """Four counts of two things in one paragraph, in two styles, with no rule to it.

    House style spells to twelve and writes figures above. A live waiver read "Twelve
    of the eighty-one sources admitted" and then, two sentences on, "the ninety
    documents the literature search reached ... : 18 documents there are leads" -- the
    same noun under the same colon written both ways, and the two spelled counts
    spelled only because the sentence one of them opened cannot start on a numeral.
    """
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import EvidencePacket, SourceLead, SourceRecord
    from coscientist.narrative import CitationRegistry

    session = Session(question="Does a coating change cycle life?")
    session.exploratory_evidence_accepted = True
    record = ResearchRecord(session=session)
    record.evidence = EvidencePacket(
        question=session.question,
        sources=[
            SourceRecord(
                url=f"https://s{index}.example/paper",
                verification_status="verified"
                if index < 15
                else "discovered_unverified",
            )
            for index in range(81)
        ],
    )
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://s{index}.example/paper", title=f"S{index}"
            )
            for index in range(104)
        ]
    )
    body = _waived_gate_advisory(record)[0].body

    assert "Of the 81 sources admitted, 15 were retrieved and checked" in body
    assert "which is not the 104 documents the literature search reached" in body
    assert "23 documents there are leads the search returned" in body
    for spelled in ("eighty-one", "fifteen", "twenty-three", "hundred"):
        assert spelled not in body, "a count above twelve is written in figures"


def test_the_waived_gate_states_the_admitted_total_where_nothing_was_checked():
    """ "That total" pointed at a total the branch above it had not printed.

    Where at least one source was retrieved the paragraph opens on the admitted count,
    and the reconciliation after it says that total is not the References total. Where
    none was, the opening sentence named no number at all and the sentence after it
    still opened "That total".
    """
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import EvidencePacket, SourceLead, SourceRecord
    from coscientist.narrative import CitationRegistry

    session = Session(question="Does a coating change cycle life?")
    session.exploratory_evidence_accepted = True
    record = ResearchRecord(session=session)
    record.evidence = EvidencePacket(
        question=session.question,
        sources=[
            SourceRecord(url=f"https://s{index}.example/paper") for index in range(3)
        ],
    )
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://s{index}.example/paper", title=f"S{index}"
            )
            for index in range(5)
        ]
    )
    body = _waived_gate_advisory(record)[0].body

    assert "No source among the three sources admitted here was retrieved" in body
    assert body.index("three sources admitted") < body.index("That total")


def test_the_waived_gate_reconciles_nothing_where_the_two_totals_agree():
    """A sentence explaining a difference, printed where there is none, invents one."""
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import EvidencePacket, SourceLead, SourceRecord
    from coscientist.narrative import CitationRegistry

    session = Session(question="Does a coating change cycle life?")
    session.exploratory_evidence_accepted = True
    record = ResearchRecord(session=session)
    record.evidence = EvidencePacket(
        question=session.question,
        sources=[SourceRecord(url="https://a.example/paper")],
    )
    record.citations = CitationRegistry(
        [SourceLead(canonical_url="https://a.example/paper", title="A")]
    )

    assert "different population" not in _waived_gate_advisory(record)[0].body


def test_the_waived_gate_names_the_approval_profile_the_cover_already_printed():
    """The regime is on the session, so the paragraph has no reason to hedge over it.

    A live report printed "Approval profile: milestone" on its cover and named the
    same profile two sections below this paragraph, and this paragraph still said
    "whichever approval regime this run used" -- the report sounding unsure of a fact
    it had stated twice already.
    """
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import ApprovalProfile, EvidencePacket, SourceRecord

    def _body(profile: ApprovalProfile) -> str:
        session = Session(question="Does a coating change cycle life?")
        session.approval_profile = profile
        session.exploratory_evidence_accepted = True
        record = ResearchRecord(session=session)
        record.evidence = EvidencePacket(
            question=session.question,
            sources=[SourceRecord(url="https://a.example/paper")],
        )
        return _waived_gate_advisory(record)[0].body

    milestone = _body(ApprovalProfile.MILESTONE)

    assert "whichever approval regime" not in milestone
    assert (
        "the milestone approval profile this run used applies to the second of "
        "those, not the first" in milestone
    )
    assert "the stage approval profile this run used" in _body(ApprovalProfile.STAGE)


def test_every_actor_this_system_records_is_named_in_words():
    """The fallback prints the internal id with its underscores taken out.

    "The waiver is recorded against the actor recorded as web researcher" stood over
    the one decision in a live report a reader is most likely to want a person for,
    and "web researcher" is a label that appears nowhere else in the document. The
    ids below are the ones the run writes for itself; an id a caller supplies is
    somebody's own name for themselves and keeps the fallback.
    """
    from coscientist.narrative import _ACTOR_WORDS, _actor_words

    for actor in (
        "cli_researcher",
        "web_researcher",
        "auto_approval_policy",
        "milestone_auto_policy",
    ):
        assert actor in _ACTOR_WORDS
        assert "recorded as" not in _actor_words(actor)
    assert _actor_words("lab_notebook_bot") == "the actor recorded as lab notebook bot"
