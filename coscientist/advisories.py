"""Every run-level warning and standing limit the report carries, in one chapter.

These paragraphs used to sit where the thing they qualify is discussed: the waived
evidence gate above the evidence, the automatic approvals in the goal section, the
templated stages under each section they touched. Each placement was defensible on
its own and the sum was not. A reader opening the overview met three paragraphs of
warning before the first sentence about the science, and prose written to be
unmissable becomes skimmable when it is the first thing on every page.

So the warnings are collected here and printed once, as the report's first appendix,
and the body carries a single line saying how many there are and how many of them
block. What is deliberately *not* here is anything specific to one hypothesis -- a
fatal flaw a reviewer recorded, or an idea's grounding verdict. Those are findings
about that idea, they belong under it, and a reader comparing two ideas has to be
able to see which one carries them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .citations import GROUNDED_STATUSES
from .narrative import (
    _AGENT_NAMES,
    IdeaBrief,
    ResearchOverview,
    ResearchRecord,
    _actor_words,
    _joined_titles,
    _listed,
    _number_word,
    _plural,
    _stage_words,
)

ADVISORY_CHAPTER = "Warnings and Limitations"
"""The appendix heading, and the name the body points at."""

AUTO_APPROVAL_WARNING = "Stage gates approved without a human"
"""Named from the provenance bullet, which used to point at the wrong section."""


def _capitalised(text: str) -> str:
    """Open a sentence with a phrase that was written to sit inside one.

    Every count in this module comes back from ``_plural`` in lower case, because
    each of these paragraphs used to begin with the word "Warning:". Without the
    prefix they open the sentence themselves.
    """
    return f"{text[:1].upper()}{text[1:]}"


@dataclass(frozen=True)
class Advisory:
    """One warning, with the heading it is filed under."""

    title: str
    body: str
    blocking: bool = False
    """Whether this one says work should not proceed, rather than qualifying it.

    The count of these is what the body's pointer leads with. A reader who is told
    only that "eight limitations apply" has no reason to turn to the appendix; a
    reader told that two of them block has every reason to.
    """


def run_advisories(
    record: ResearchRecord,
    *,
    overview: ResearchOverview | None = None,
    briefs: tuple[IdeaBrief, ...] = (),
) -> list[Advisory]:
    """Everything that qualifies this report as a whole, worst first.

    Ordered by consequence rather than by the stage that raised it, because the
    appendix is read top-down by someone deciding whether to act on the report.
    """
    advisories = [
        *_governance_advisory(record),
        *_broken_grounding_advisory(briefs),
        *_waived_gate_advisory(record),
        *_templated_stage_advisory(record),
        *_stood_in_review_advisory(record),
        *_mechanical_overview_advisory(overview),
        *_automatic_approval_advisory(record),
        _standing_limits_advisory(record),
    ]
    return advisories


def advisory_pointer(advisories: list[Advisory]) -> str:
    """The one line the body carries in place of the paragraphs moved out of it.

    It leads with the count of blocking items rather than the total, because the
    total is the same on a clean run as on a halted one and gives a reader no reason
    to turn the page.
    """
    blocking = [item for item in advisories if item.blocking]
    opening = (
        _capitalised(_plural(len(advisories), "limitation"))
        + (" applies" if len(advisories) == 1 else " apply")
        if advisories
        else "No limitation applies"
    )
    if not blocking:
        return (
            f"{opening} to this report as a whole, and "
            + ("it does not block" if len(advisories) == 1 else "none of them blocks")
            + " the work proposed below. "
            + ("It is" if len(advisories) == 1 else "They are")
            + f" set out under {ADVISORY_CHAPTER} at the end."
        )
    return (
        f"{opening} to this report as a whole, of which "
        + (
            "one says the work should not proceed on the material it names"
            if len(blocking) == 1
            else f"{_number_word(len(blocking)).lower()} say the work should not "
            "proceed on the material they name"
        )
        + f": {_listed([item.title.lower() for item in blocking])}. "
        f"All are set out under {ADVISORY_CHAPTER} at the end, and nothing here "
        "should be acted on without reading them."
    )


# ---------------------------------------------------------------------------
# The individual advisories
# ---------------------------------------------------------------------------


def _governance_advisory(record: ResearchRecord) -> list[Advisory]:
    if not record.open_governance_blocks:
        return []
    return [
        Advisory(
            title="Unanswered fatal governance findings",
            blocking=True,
            body=(
                _capitalised(
                    _plural(
                        len(record.open_governance_blocks), "fatal governance finding"
                    )
                )
                + " in this run "
                f"{'has' if len(record.open_governance_blocks) == 1 else 'have'} not "
                "been answered by anyone, covering "
                + _joined_titles(
                    sorted({item.title for item in record.open_governance_blocks}),
                    fallback="no idea",
                )
                + ". The affected work is blocked, not approved, and this report "
                "should not be used to justify starting it."
            ),
        )
    ]


def _broken_grounding_advisory(briefs: tuple[IdeaBrief, ...]) -> list[Advisory]:
    # The two verdicts collected here break in different ways, and this paragraph
    # used to assert the unsupported reason -- "resolve to no record" -- over both.
    # A discredited idea's citations do resolve; the record they resolve to was
    # retracted or could not be retrieved. On a live report that meant four ideas
    # were told here that nothing they cite exists, and told under their own
    # headings that the source is Small changes, big gains and its claim is
    # discredited. Both cannot be true, and the reader had no way to pick.
    absent = [brief for brief in briefs if brief.support == "unsupported"]
    retracted = [brief for brief in briefs if brief.support == "discredited"]
    if not absent and not retracted:
        return []
    findings = []
    where = " in this report"
    if absent:
        one = len(absent) == 1
        findings.append(
            f"{_capitalised(_plural(len(absent), 'idea'))}{where} "
            f"{'cites' if one else 'cite'} evidence that does not exist "
            "in this session, namely "
            + _joined_titles([brief.title for brief in absent], fallback="none")
            + ". "
            + (
                "That citation was written by the generator and resolves to no "
                "record, so the idea carrying it is unsupported rather than "
                "evidence-backed, whatever its rank says."
                if one
                else "Those citations were written by the generator and resolve to "
                "no record, so the ideas carrying them are unsupported rather than "
                "evidence-backed, whatever their rank says."
            )
        )
        where = ""
    if retracted:
        one = len(retracted) == 1
        findings.append(
            f"{_capitalised(_plural(len(retracted), 'idea'))}{where} "
            f"{'cites' if one else 'cite'} evidence that was retracted "
            "or that this run could not retrieve, namely "
            + _joined_titles([brief.title for brief in retracted], fallback="none")
            + ". "
            + (
                "That citation does resolve to a record, and the record no longer "
                "stands, so the idea carrying it is discredited rather than "
                "evidence-backed, whatever its rank says."
                if one
                else "Those citations do resolve to records, and the records no "
                "longer stand, so the ideas carrying them are discredited rather "
                "than evidence-backed, whatever their rank says."
            )
        )
    if not retracted:
        title = "Ideas citing evidence that does not exist"
    elif not absent:
        title = "Ideas citing evidence that has been retracted"
    else:
        title = "Ideas citing evidence that is absent or retracted"
    return [
        Advisory(
            title=title,
            blocking=True,
            body=(
                " ".join(findings) + " Nothing in this group should be acted on "
                "until its grounding is rebuilt."
            ),
        )
    ]


def _corpus_reconciliation(record: ResearchRecord, admitted: int) -> str:
    """Why this paragraph's total is not the total the References prose gives.

    Section eight counts the documents the literature search reached and this counts
    the sources the evidence stage admitted. On a live run they were eighty and
    seventy-seven, both introduced by the same "twenty-five of the", four chapters
    apart, with nothing anywhere saying they are counts of different things -- so the
    reader's only available reading was that one of the two is wrong.
    """
    registry = record.citations
    gathered = registry.verification_standing[1]
    if not gathered or gathered == admitted:
        return ""
    sources = record.evidence.sources if record.evidence else []
    # Through the fold, because two admitted sources can be one document, and a
    # source the packet recorded without a link is no document at all.
    documents = {
        document
        for source in sources
        if (document := registry.document_for(source.url or ""))
    }
    unnumbered = len(sources) - len(documents)
    unadmitted = gathered - len(documents)
    stated = [
        f"{_plural(unadmitted, 'document')} there "
        + ("is a lead" if unadmitted == 1 else "are leads")
        + " the search returned that no admitted source names"
        if unadmitted > 0
        else "",
        f"{_plural(unnumbered, 'source')} admitted here "
        + ("names" if unnumbered == 1 else "name")
        + " no document that can be numbered"
        if unnumbered > 0
        else "",
    ]
    difference = _listed([item for item in stated if item])
    return (
        "That total counts what the evidence stage admitted, which is not the "
        f"{_number_word(gathered).lower()} documents the literature search reached "
        "that Key Findings counts" + (f": {difference}. " if difference else ". ")
    )


def _waived_gate_advisory(record: ResearchRecord) -> list[Advisory]:
    if not record.session.exploratory_evidence_accepted:
        return []
    # "The operator explicitly accepted limited exploratory evidence" once sat two
    # paragraphs below a warning that every gate in the run, this one included, had
    # been approved automatically with nobody reading the artifact. Both are true of
    # different acts -- the waiver was requested, the acceptance was granted by the
    # profile -- but stated side by side without that distinction they read as one
    # claim contradicting itself. The actor is on the event, so it is named rather
    # than characterised.
    waiver = next(
        (
            event
            for event in reversed(record.session.events)
            if event.event_type == "limited_exploratory_evidence_accepted"
        ),
        None,
    )
    # "Nothing among them should be cited as established" was written for a waiver
    # granted over an unverified corpus, and it was printed over one where fifteen of
    # fifty-nine sources had been retrieved and checked and were badged
    # "[Verified Source]" throughout the body. The waiver means the corpus as a whole
    # fell short of the declared standard, not that every source in it is a lead, and
    # the sentence has to say which of the two the reader is holding.
    sources = record.evidence.sources if record.evidence else []
    grounded = [
        source for source in sources if source.verification_status in GROUNDED_STATUSES
    ]
    return [
        Advisory(
            title="A waived evidence gate",
            blocking=True,
            body=(
                "The evidence gate for this run was waived, so the literature under "
                "Knowledge Base was admitted without meeting the verification "
                "standard the goal declared. "
                + (
                    f"The waiver is recorded against {_actor_words(waiver.actor)}. "
                    if waiver
                    else "No actor is recorded against the waiver. "
                )
                + "Waiving the gate is a distinct act from accepting the stage's "
                "output, and whichever approval regime this run used applies to the "
                "second of those, not the first. "
                + (
                    # Both counts are spelled, because "five sources of the 8
                    # admitted" is one sentence written in two number styles.
                    f"{_number_word(len(grounded))} of the "
                    f"{_number_word(len(sources)).lower()} sources admitted "
                    + ("was" if len(grounded) == 1 else "were")
                    + " retrieved and checked against the claim drawn from "
                    + ("it" if len(grounded) == 1 else "them")
                    + ", and "
                    + (
                        "that one is marked as verified where it is cited"
                        if len(grounded) == 1
                        else "those are marked as verified where they are cited"
                    )
                    + ". The rest are "
                    "exploratory leads rather than findings and should not be cited "
                    "as established. "
                    if grounded
                    else "No source admitted here was retrieved and checked, so all "
                    "of them are exploratory leads rather than findings and none "
                    "should be cited as established. "
                )
                + _corpus_reconciliation(record, len(sources))
                + "The gate should be re-run before any idea grounded on it is acted "
                "upon."
            ),
        )
    ]


def _templated_stage_advisory(record: ResearchRecord) -> list[Advisory]:
    stages = record.fallback_stages
    if not stages:
        return []
    # "the reflection agent output" and "failed contract validation" are the run's
    # own words for itself. The reader is being told not to trust a page of the
    # report, which is the last place to make them decode an agent id.
    named = sorted(
        {_AGENT_NAMES.get(note.agent, note.agent.replace("_", " ")) for note in stages}
    )
    agents = _listed(named, fallback="one specialist").rstrip(".")
    # Only the verb used to agree with the count, so three stages that fell back at
    # once were "the clustering by mechanism, evolution of the shortlist, and
    # meta-review are a fixed template, not the specialist's own reasoning" -- one
    # template and one specialist between three named stages.
    one = len(named) == 1
    return [
        Advisory(
            title="Stages that produced a template rather than reasoning",
            body=(
                f"The {agents} in this report "
                + (
                    "is a fixed template, not the specialist's own reasoning. The "
                    "specialist's answer came back incomplete or malformed and was "
                    "replaced"
                    if one
                    else "are fixed templates, not the specialists' own reasoning. "
                    "Their answers came back incomplete or malformed and were "
                    "replaced"
                )
                + ", so what "
                + ("it" if one else "they")
                + " contributed to this report states what the workflow requires "
                "rather than what a model concluded. Treat it as a placeholder and "
                "re-run the "
                + ("stage" if one else "stages")
                + " before relying on it. What each stage produced is itemised under "
                "Provenance."
            ),
        )
    ]


def _stood_in_review_advisory(record: ResearchRecord) -> list[Advisory]:
    """A review nobody wrote, printed under an idea as though somebody had.

    Filed next to the templated-stage warning above because it is the same substitution
    at a smaller grain, and the stage-level warning does not catch it: a reviewer that
    answers for seven of eight ideas has its stage recorded as the specialist's own.
    On a live run the one backfilled review landed on the rank-1 idea, carried the
    lowest score that idea received, and the conclusion under it sent the reader to
    that review as the one to read before commissioning the work.
    """
    stood = record.stood_in_reviews
    if not stood:
        return []
    one = len(stood) == 1
    affected = _joined_titles(
        sorted({record.ranked_title(review.candidate_id) for review in stood}),
        fallback="no idea",
    )
    return [
        Advisory(
            title="Reviews that no reviewer wrote",
            body=(
                _capitalised(_plural(len(stood), "review"))
                + " printed in this report "
                + ("was" if one else "were")
                + " filled in from a fixed template rather than written by a reviewer, "
                + ("on " if one else "across ")
                + affected
                + ". A reviewer answered for some of the ideas and not others, and the "
                "run backfilled the rest so that every idea carries the same sections. "
                + ("The placeholder states" if one else "The placeholders state")
                + " nothing about the "
                + ("idea it sits" if one else "ideas they sit")
                + " under, but "
                + ("its verdict and score" if one else "their verdicts and scores")
                + " entered the score spreads, the averages and the ranking as though "
                "a reviewer had set them down. Each is named where it is printed. Any "
                "idea named here is unreviewed on that criterion rather than weak on "
                "it, and the criterion should be re-run before the ordering is relied "
                "upon."
            ),
        )
    ]


def _mechanical_overview_advisory(
    overview: ResearchOverview | None,
) -> list[Advisory]:
    if overview is None or overview.source != "deterministic_fallback":
        return []
    return [
        Advisory(
            title="A mechanically assembled overview",
            # "the report compiler", "a synthesis specialist" and "a recorded field"
            # are three names for parts of this system, and a reader has met none of
            # them. What the warning has to convey is simpler and does not need them:
            # a program wrote it, so nothing in it is a judgement formed by reading
            # the run.
            body=(
                "Research Overview was assembled mechanically from what each stage of "
                "the run recorded. Every sentence in it restates one of those "
                "records, and no model was asked to read the run as a whole, so where "
                "two stages disagree the disagreement is reported rather than "
                "resolved."
            ),
        )
    ]


def _automatic_approval_advisory(record: ResearchRecord) -> list[Advisory]:
    session = record.session
    # Acceptances only. A gate is closed by an acceptance, and counting every
    # automatic decision would let a revision the policy recorded stand in this
    # warning as a stage nobody read.
    automatic = [
        decision
        for decision in session.decisions
        if decision.automatic and decision.action == "accept"
    ]
    if not automatic:
        return []
    return [
        Advisory(
            title=AUTO_APPROVAL_WARNING,
            body=(
                f"{_capitalised(_plural(len(automatic), 'stage gate'))} in this run "
                f"{'was' if len(automatic) == 1 else 'were'} approved automatically "
                f"under the {session.approval_profile} approval profile, covering "
                + _listed(
                    _stage_words(decision.stage for decision in automatic),
                    fallback="no stage",
                )
                # "No human inspected the artifact" and "the payload satisfied its
                # contract" are three words of implementation vocabulary in two
                # sentences, in the one warning a reader most needs to act on.
                + ". Nobody read what "
                + ("that stage" if len(automatic) == 1 else "those stages")
                + " produced before it was accepted. An acceptance recorded here "
                "means only that the work was complete "
                "and well-formed, not that a person agreed with it. Auto approval is "
                "a workflow convenience and never constitutes scientific, safety, "
                "ethics, or institutional approval."
            ),
        )
    ]


def _standing_limits_advisory(record: ResearchRecord) -> Advisory:
    """True of every run, and therefore the one advisory that is never absent.

    It reads as boilerplate because it is boilerplate, which is exactly the argument
    for having it here rather than in the opening paragraph of the overview, where it
    was the fourth sentence a reader ever read.
    """
    session = record.session
    return Advisory(
        title="What this report is and is not",
        body=(
            "Nothing in this document is a finding. The ideas are proposals that have "
            "been reviewed, ranked and stress-tested against each other, and each "
            "still requires independent verification before it is acted upon. A "
            "source satisfies an evidence gate only when its original content has "
            "been inspected and mapped to the exact claim.\n\n"
            + (
                "The run was executed as a literature-only analysis. "
                if session.literature_only
                else ""
            )
            + "No experiment was performed in this run, no dataset was accessed and "
            "no measurement was taken. Every stage of it is desk work: a literature "
            "search, a set of proposals written by models, and reviews of those "
            "proposals by other models. So every quantitative statement in the report "
            "is an expectation recorded by whichever specialist proposed it, and the "
            "research mode named on the cover"
            + (
                f", {session.research_mode.replace('_', ' ')},"
                if session.research_mode
                else ""
            )
            + " describes the work being proposed rather than any work that was done."
        ),
    )
