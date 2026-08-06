"""Narrative synthesis that turns accepted artifacts into a readable research report.

The dossier used to be an artifact dump: opaque candidate ids, raw JSON payloads and
one section per pipeline stage. The reference reports this project is modelled on read
as a scientific document instead, so this module builds the intermediate layer that
was missing — human titles for every idea, a dense citation registry over the
discovery manifest's source leads, per-idea deep-dive briefs, and the nine narrative
sections themselves. It holds no formatting logic; ``dossier.compile_dossier`` renders
the structures produced here into Markdown.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any, Literal

from pydantic import Field

from .citations import GROUNDED_STATUSES, CandidateCitations, resolve_population
from .evidence import GROUNDING_REDIRECT_MARKER
from .governance import open_blockers
from .models import (
    EVIDENCE_FACETS,
    FACET_PHRASES,
    STAGES,
    ArtifactStatus,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    Contract,
    DiscoveryManifest,
    DiscoveryNarrative,
    DossierManifest,
    EvidencePacket,
    EvolutionCycle,
    EvolutionRecord,
    ResearchCluster,
    ResearchLandscape,
    ResearchPlan,
    ReviewSet,
    Session,
    SourceLead,
    TournamentState,
)
from .parity import DEFAULT_ELO, ELO_K, UNMEASURED_MOVEMENT

# The four evidence-quality qualifiers the reference reports attach to citations.
# Anything outside this set is a formatting bug, so the renderer and tests share it.
CITATION_ANNOTATIONS = (
    "leaning accurate",
    "inaccurate",
    "disputed",
    "unsupported",
)

# At most a quarter of citation groups may carry a qualifier. Past that the
# annotations stop reading as exceptions and start reading as decoration, which is
# the opposite of what they are for.
CITATION_ANNOTATION_CEILING = 0.25

# The reference reports collect every review under four scored section names, then
# close with two sections that carry no score. Bucketing by criterion rather than by
# reviewer matters because two reviewers share the impact criterion, and per-reviewer
# headings would print the same title twice inside one idea.
REVIEW_SECTIONS = ("Correctness", "Novelty", "Feasibility", "Impact", "Safety")
CRITERION_SECTIONS = {
    "evidence_correctness": "Correctness",
    "novelty": "Novelty",
    "methods_feasibility": "Feasibility",
    "impact_safety": "Impact",
    # Safety decides whether work may proceed at all and impact decides whether it
    # is worth proceeding, so they are read separately. Filing a safety review under
    # another heading is how a hazard becomes a footnote.
    "safety_governance": "Safety",
}
UNSCORED_REVIEW_SECTIONS = ("Coherence", "Deep Verification")

# The workflow order, as a rank, for sorting records that were written out of it.
_STAGE_ORDER = {stage: index for index, stage in enumerate(STAGES)}

# The two convergence reasons that mean the Deep Research agent produced nothing and
# something else stood in for it. Four passages across the narrative and the appendix
# turn on that fact, and each of them used to carry its own copy of the pair: a reason
# added to one and not the others would have made the report disagree with itself
# about whether the designed path for the evidence stage ever ran.
DISCOVERY_STOOD_IN = frozenset(
    {"search_grounded_fallback", "deep_research_unavailable"}
)

# The eight fixed subsections of a per-idea Summary, in the reference reports' order.
SUMMARY_SUBSECTIONS = (
    "Executive Verdict",
    "Critical Flaws",
    "Identified issues & Validated Risks",
    "Addressed Objections",
    "Supporting Arguments & Evidence (Motivation)",
    "Goal Alignment & Novelty",
    "Feasibility Assessment (Go/No-Go Decision)",
    "Conclusion",
)

# Recommendations map onto the reference reports' 1-5 review scale. Keeping the
# mapping explicit means a reader can invert it rather than guess what a 4 meant.
RECOMMENDATION_SCORES = {
    "advance": 5,
    "revise": 3,
    "insufficient_evidence": 2,
    "reject": 1,
}

# One consistent row-label set per report, as in the reference documents: the table
# is a comparison grid, so the labels cannot vary between ideas.
IDEA_TABLE_ROWS = (
    "Mechanism",
    "Discriminating prediction",
    "Falsifier",
    "Key dependency",
    "Principal risk",
)


class NarrativeSubsection(Contract):
    """A numbered subsection such as ``4.3``, optionally carrying the idea grid."""

    number: str
    title: str
    paragraphs: list[str] = Field(default_factory=list)
    table_rows: list[tuple[str, str]] = Field(default_factory=list)


class NarrativeGrid(Contract):
    """A grid a section prints where a list would otherwise run as prose.

    Eight ideas with a rating and a win-loss record apiece is a table, and written out
    as a sentence -- "the full standings run A at 1290 (6-0); B at 1234 (4-2); ..." --
    it is ninety words a reader has to parse serially to compare two rows of it.
    """

    after: int = Field(ge=0)
    """The index in ``paragraphs`` this grid follows.

    Elaboration paragraphs are appended after the core ones are seated, so an index
    into the core list stays valid however the word budget fills the section out.
    """
    columns: list[str]
    rows: list[list[str]]


class NarrativeSection(Contract):
    """One numbered section of the report's narrative body."""

    number: int = Field(ge=1, le=9)
    title: str
    paragraphs: list[str] = Field(default_factory=list)
    subsections: list[NarrativeSubsection] = Field(default_factory=list)
    grids: list[NarrativeGrid] = Field(default_factory=list)


class ResearchOverview(Contract):
    """The written report layered over the typed artifacts.

    A specialist may produce this directly; when none has, it is assembled from the
    accepted artifacts instead. ``source`` records which happened, because a reader
    must be able to tell a synthesised narrative from a derived one.
    """

    goal_title: str
    report_title: str
    sections: list[NarrativeSection] = Field(default_factory=list)
    research_directions: list[str] = Field(default_factory=list)
    review_summary: list[str] = Field(default_factory=list)
    knowledge_summary: str = ""
    open_questions: list[str] = Field(default_factory=list)
    open_questions_lead_in: str = ""
    """Where the evidence that would change the recommendation is stated instead."""
    unexpected_connections: list[str] = Field(default_factory=list)
    connections_lead_in: str = ""
    """What the cross-links are, since the reference heading over them does not say."""
    source: Literal["specialist", "deterministic_fallback"] = "specialist"

    @property
    def word_count(self) -> int:
        return sum(
            len(paragraph.split())
            for section in self.sections
            for paragraph in _section_prose(section)
        )


def _section_prose(section: NarrativeSection) -> list[str]:
    """Every paragraph a section prints, subsections included, for the word budget."""
    return list(section.paragraphs) + [
        paragraph
        for subsection in section.subsections
        for paragraph in subsection.paragraphs
    ]


@dataclass(frozen=True)
class Citation:
    """A numbered reference. Deep Research returns redirect URLs, so the title leads."""

    number: int
    title: str
    url: str


class CitationRegistry:
    """Assigns reference numbers on first use so the sequence is dense from 1.

    Numbering by position in the manifest would leave gaps wherever a lead is never
    cited, and a reference list with holes reads as a rendering fault.
    """

    def __init__(
        self,
        leads: Sequence[SourceLead],
        *,
        annotations: dict[str, str] | None = None,
    ) -> None:
        self._leads = {lead.canonical_url: lead for lead in leads if lead.canonical_url}
        self._annotations = annotations or {}
        self._numbers: dict[str, int] = {}
        self._ordered: list[str] = []
        self._groups = 0
        self._annotated = 0
        self._universal = self._uniform_qualifiers()

    def _uniform_qualifiers(self) -> set[str]:
        """Qualifiers true of every source, which are therefore not qualifiers.

        A tag beside a citation says: this source is unlike the others. On a run where
        nothing was verified, every source is "unsupported" -- and the ceiling below
        then prints the tag on a quarter of them, which tells a reader the untagged
        three quarters were checked. Nothing was. A fact true of the whole reference
        list belongs in the prose that introduces the list, stated once.
        """
        if not self._leads:
            return set()
        found = {self._annotations.get(url, "") for url in self._leads}
        # A source carrying no verification record and one recorded as "unsupported"
        # are the same fact stated two ways: nothing established the document behind
        # it. Reading the empty string as a qualifier in its own right is what silenced
        # the lead-in on a run where verification recorded nothing whatever -- fifty
        # three leads, no verdicts, and a reference list that said nothing about
        # whether any of them had been checked.
        if found <= {"", "unsupported"}:
            return {"unsupported"}
        return found if len(found) == 1 else set()

    @property
    def verification_standing(self) -> tuple[int, int]:
        """How many sources available to cite were checked against their document.

        Counted over every lead the registry can number rather than over the entries
        that ended up cited, because the reference list is settled after the prose
        describing it is written.
        """
        checked = sum(
            1
            for lead in self._leads.values()
            if lead.verification_status in GROUNDED_STATUSES
        )
        return checked, len(self._leads)

    @property
    def universal_qualifier(self) -> str:
        """The qualifier that holds of every cited source, if one does.

        ``_uniform_qualifiers`` withholds a qualifier from the citations themselves on
        the stated grounds that a fact true of the whole list belongs in the prose that
        introduces it. No such prose existed, so on a run where nothing was verified
        the report simply stopped saying so anywhere near the sources.
        """
        return next(iter(self._universal), "")

    def __len__(self) -> int:
        return len(self._ordered)

    @property
    def annotation_rate(self) -> float:
        """Share of emitted citation groups carrying an evidence qualifier."""
        return self._annotated / self._groups if self._groups else 0.0

    def number(self, url: str) -> int | None:
        """Resolve a URL to its reference number, assigning one on first sight."""
        if url not in self._leads:
            return None
        if url not in self._numbers:
            self._ordered.append(url)
            self._numbers[url] = len(self._ordered)
        return self._numbers[url]

    def marker(self, urls: Iterable[str], *, annotate: bool = True) -> str:
        """Render ``[1, 2]`` plus at most one evidence qualifier for those sources."""
        numbers = sorted({n for url in urls if (n := self.number(url)) is not None})
        if not numbers:
            return ""
        marker = "[" + ", ".join(str(number) for number in numbers) + "]"
        if not annotate:
            return marker
        self._groups += 1
        qualifier = next(
            (
                self._annotations[url]
                for url in urls
                if self._annotations.get(url) in CITATION_ANNOTATIONS
            ),
            "",
        )
        # A qualifier states something true about the source, so it is never invented
        # to reach a quota; it is only withheld once the page is dense with them.
        if not qualifier or qualifier in self._universal:
            return marker
        if self._annotated + 1 > CITATION_ANNOTATION_CEILING * self._groups:
            return marker
        self._annotated += 1
        return f"{marker} ({qualifier})"

    def references(self) -> list[Citation]:
        """The cited leads only, in citation order, titled rather than linked."""
        return [
            Citation(
                number=index,
                title=_reference_title(self._leads[url]),
                url=url,
            )
            for index, url in enumerate(self._ordered, start=1)
        ]


_HOSTNAME_TITLE = re.compile(r"^(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}$", re.IGNORECASE)


def _untitled_on(host: str) -> str:
    """What to call a source the search captured no title for."""
    return f"Untitled source on {host}" if host else "Untitled source lead"


def _publisher_of(url: str) -> str:
    """The host a locator points at, where that host is a publisher and not a hop."""
    if not url.startswith(("http://", "https://")) or GROUNDING_REDIRECT_MARKER in url:
        return ""
    _, _, remainder = url.partition("://")
    host, _, _ = remainder.partition("/")
    return host.removeprefix("www.")


# A search result's title, as the grounding API hands it over, ends in the site it
# was found on: "... by atomic layer deposition - The Royal Society of Chemistry
# rsc.org". The hostname is the search engine's own furniture. Printed in a
# reference list it reads as part of the paper's name, and the same paper found on
# two aggregators then carries two different names.
_TRAILING_HOST = re.compile(
    r"[\s,;|/\u2013\u2014-]+(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,}\.?$",
    re.IGNORECASE,
)


# What the aggregators put in front of the hostname, and what is left behind once
# it goes. "Corrosion-inspired design of artificial interphases ... - DOI doi.org"
# reached a live review as the name of a paper; cutting "doi.org" leaves "- DOI",
# which is no better. The words listed are labels for a link, never the last words
# of a paper's title.
_TRAILING_LINK_LABEL = re.compile(
    r"[\s,;|/\u2013\u2014-]+(?:doi|pdf|html|pmc|abstract|full[\s-]?text|"
    r"download|view|open access)\.?$",
    re.IGNORECASE,
)


def _without_search_chrome(title: str) -> str:
    """A search result's title without the link furniture appended to it."""
    trimmed = title
    for _ in range(4):
        cut = _TRAILING_LINK_LABEL.sub("", _TRAILING_HOST.sub("", trimmed)).strip(
            " .,;|-\u2013\u2014"
        )
        if cut == trimmed:
            break
        trimmed = cut
    # Only where something recognisable as a title is left. On a result whose whole
    # title is the hostname there is nothing to keep, and the caller says so instead.
    return trimmed if len(trimmed.split()) >= 3 else title


def _reference_title(lead: SourceLead) -> str:
    """Prefer the annotation title: the canonical URL is a grounding redirect."""
    title = _without_search_chrome(" ".join(lead.title.split()))
    if not title:
        # A lead with no title still usually has a link, and the link names the
        # publisher. A live list printed "Untitled source lead." -- a reference
        # saying nothing whatever about what it referred to -- next to a locator
        # that said acs.org. Only a lead that has neither falls through to it now.
        title = _untitled_on(_publisher_of(lead.canonical_url))
    elif _HOSTNAME_TITLE.match(title):
        # Discovery falls back to the hostname when the search result carries no
        # title, so the reference list printed entries reading "www.mdpi.com" as
        # though that were the name of a paper. It is a publisher, and saying which
        # publisher and that the title is missing is both shorter and true.
        title = _untitled_on(title.removeprefix("www."))
    if lead.year:
        title = f"{title} ({lead.year})"
    return title


@dataclass(frozen=True)
class IdeaReview:
    """One review rendered as the reference reports do, ending in a matched score."""

    section: str
    lead_in: str
    question: str
    findings: list[str]
    objections: list[str]
    rebuttals: list[str]
    answer: str
    score: int
    recommendation: str = ""
    """What the reviewer asked for, kept unworded so agreement can be counted."""
    fatal_flaws: list[str] = field(default_factory=list)
    """The findings the reviewer judged disqualifying, as distinct from its objections.

    Carried through separately because the two are answerable in different ways and
    the report has to say which it is holding. Omitting the field here is what kept
    the flaws out of the document: the review that recorded one arrived at the page
    as a score of two with a list of objections that did not include it.
    """


@dataclass(frozen=True)
class IdeaMatch:
    """A single tournament pairing, including the debate transcript when there is one."""

    round_number: int
    opponent_title: str
    outcome: str
    elo_before: float
    elo_after: float
    confidence: float
    rationale: str
    judge: str
    debate_turns: list[str] = field(default_factory=list)
    unreadable_turns: int = 0
    # The rating this row opens at, as printed. It is chained from the first row the
    # idea played rather than rounded from this row's own fractional endpoint: with
    # per-row rounding a row closed at 1290 and the next opened at 1289, and the
    # headline Elo -- a third rounding, of the final rating -- agreed with neither.
    # A reader has no way to read a table that does not add up as anything but an
    # error, so the printed arithmetic is made the authority and the headline is
    # taken from where the table ends. Set by ``_idea_matches``.
    shown_before: int = 0

    @property
    def swing(self) -> int:
        """What this match moved the rating by, rounded once rather than twice.

        Elo moves the two sides of a decided match by exactly opposite amounts, but
        the endpoints are fractional and rounding each of them independently breaks
        that symmetry: one live table showed the winner of a match gaining 13 points
        and its opponent losing 14 in the same row of the opposing table. Rounding
        the difference instead of differencing the roundings restores it.
        """
        return round(self.elo_after - self.elo_before)

    @property
    def shown_after(self) -> int:
        """The endpoint the printed arithmetic reaches, not a third rounding of it."""
        return self.shown_before + self.swing


# The chapter the per-idea sections are printed under, named here so a cross-reference
# cannot drift from the heading it points at. Section 8 used to send the reader to
# "1. The Primary Mechanism of Cycle Life Extension above": a numbered heading this
# report does not print, since the deep dives are headed by title alone, and in a
# direction the reader cannot follow, since they sit after the overview rather than
# before it. A pointer that misnames its target and then misdirects the reader is worse
# than no pointer at all.
DEEP_DIVE_CHAPTER = "Top ideas in detail"


# The conventions the deep dives are read under, stated once above all of them. Each
# of these qualifications applies to every idea in the report, so printed per idea they
# cost the same words eight times over -- and a reader who has met a caveat three times
# stops reading it, which is the opposite of what a caveat is for.
DEEP_DIVE_PREAMBLE = (
    "Every idea below is set out the same way, so that a section can be compared "
    "across ideas rather than read only in place. What the fixed sections hold, and "
    "the qualifications that hold for every idea in this report, are stated here "
    "once rather than under each.",
    "Description states the idea in the proposing specialist's own terms and in a "
    "fixed order: the mechanism it rests on, the predictions that separate it from "
    "its neighbours, the competing reading of the same situation, and the "
    "observation that would falsify it. A mechanism that does not hold takes "
    "everything under it with it; a prediction every competing idea also makes "
    "distinguishes nothing; and a falsifier stated before the work starts is what "
    "keeps the idea a hypothesis rather than a position.",
    "Evidence Assessment is the proposing specialist's own reading of the "
    "literature it worked from: what it takes to argue for the idea, what it takes "
    "to argue against it, and what it already knew was missing. Each statement "
    "carries what stands behind it — **[Verified Source]** where it names a "
    "document this run retrieved and checked, **[Literature Lead]** where it names "
    "one the search found but nothing confirmed, and **[Unsourced claim]** where it "
    "names no document at all. A gap carries no label, because a statement that no "
    "evidence exists is not one that can be grounded.",
    "Identified issues and validated risks are the risks the specialist that "
    "proposed the idea named against its own work. No reviewer was asked to confirm "
    "them, so the list is neither validated nor complete, and a risk missing from it "
    "has not been ruled out.",
    "Addressed objections are the responses the reviews recorded. A review lists its "
    "objections and its responses separately without saying which answers which, so "
    "a response may concede an objection rather than dispose of it. Each response is "
    "attributed to the review that recorded it, so that is where a response is "
    "printed rather than a second time under the review itself, and a review that "
    "recorded none is not listed there rather than listed as silent. The objections "
    "themselves are under Deep Verification.",
    "Supporting arguments state what the idea's mechanism predicts rather than the "
    "mechanism again, since a prediction specific enough to come out one way is what "
    "separates an idea worth running from one merely worth believing. Where findings "
    "are listed there, they are the ones the idea's own proposal cites; the evidence "
    "stage classified each of them for or against the research question and not "
    "against any one idea, so a finding can be listed under an idea it in fact tells "
    "against. No prediction anywhere in this report has been tested: the run proposes "
    "work rather than carrying any out, so a prediction is a case for the idea rather "
    "than a result of it. Goal alignment "
    "and novelty is a score and no more: what an idea has to displace is the "
    "competing reading set out under its own Description, and the score is a "
    "judgement about that contest rather than about the idea in isolation.",
    # What the go/no-go tests are for framed all eight lists of them, at fourteen
    # words a list, above the tests themselves.
    "The feasibility assessment gives the review's score, what has to exist before "
    "the idea can be started at all, and the go/no-go tests. Those tests are what "
    "continuing or abandoning is decided against, and the specialist set them down "
    "before the work began, which is what keeps the decision from being made on the "
    "result once it is in hand.",
    # Why the go/no-go precedes the falsifier is the same argument under every idea
    # that recorded both, and it was the opening twenty words of eight conclusions.
    "The conclusion under each idea names the next move on it. Where an idea records "
    "both a go/no-go check and a falsifier, the go/no-go comes first: it is the "
    "cheaper of the two, and a failure there ends the work without the falsifier ever "
    "being run.",
    # Printed under each of the four ideas that had nothing to report there, which is
    # exactly the set of ideas whose reader most needs it and exactly four copies.
    # The heading is Revised Form Recommended only under an idea the meta-review
    # carries, so the preamble names the part of it every rewritten idea has.
    "Where an idea carries a Revised Form section, only what the rewrite changed is "
    "set out in it: the fields that carry over unchanged are named there rather than "
    "printed a second time, and they are to be read from the idea above. Evolution "
    "rewrites the whole shortlist, so the heading reads Revised Form Recommended only "
    "where the meta-review went on to recommend the rewrite.",
    "Where Critical Flaws reports that no reviewer recorded a fatal flaw, that is a "
    "statement about the reviews rather than a guarantee: a flaw nobody looked for is "
    "indistinguishable here from one that does not exist.",
    # Where the objections are printed was also the tail of every review's own
    # objection line -- five reviews to an idea and eight ideas, so forty copies of
    # one signpost. The count is what varies review to review; the destination is
    # not, and it is already half-stated two paragraphs above.
    "Deep Verification lists the fatal flaws and the objections the reviews raised, "
    "each numbered and attributed to the review that raised it, and all of them to "
    "be checked rather than argued away. Nothing in this run tested any item in those "
    "lists, so each is a live claim against the idea it is filed under rather than a "
    "settled one. A fatal flaw is the stronger "
    "finding: the reviewer that recorded one was saying the idea does not survive it, "
    "and the scoring scale caps such a review at two of five however confident it was. "
    "For the same reason no item there claims to have been answered: where a review "
    "responded at all, the responses are printed under Reviews, and which objection "
    "each one reaches is left to the reader to judge.",
    # A debated match has two participants and each of them has a section, so the
    # exchange is printed twice and a reader meeting the second copy takes it for an
    # editing fault. It is not one -- the debaters argue in slots, and each printing
    # resolves the slots to "this idea" and "the opposing idea" from its own section's
    # side, so the two copies read as opposite arguments. Saying so once is cheaper
    # than either abbreviating the loser's section or leaving the reader to work it
    # out from a transcript that appears to contradict the one they just read.
    "A debate appears under both of the ideas that fought it, each time argued from "
    "that section's side: the same exchange, read from the opposite end. The verdict "
    "and the rating change under it are the same in both places. A judge argues its "
    "verdict in the closing turn of the exchange, so where the verdict under a debate "
    "carries nothing further, its reasoning is that closing turn rather than missing; "
    "where the judge reasoned somewhere other than in the exchange, the verdict "
    "carries that reasoning, and where it recorded none the verdict says so.",
)

# What introduces an idea's check list when the list came from its reviews. It is the
# default an idea brief carries, and it sits here rather than beside the function that
# builds the list because a dataclass default has to exist before the class does.
# No lead-in where the list is what the preamble above the ideas already says it is.
# Each item names the review that raised it and whether it is a flaw or an objection,
# so a sentence saying the list holds flaws and objections restated the items under it
# and the preamble above it, once per idea, seven times on a live run. What survives
# is the case the preamble cannot state: the one where nothing was raised at all.
DEEP_VERIFICATION_LEAD_IN = ""
DEEP_VERIFICATION_FATAL_LEAD_IN = ""


@dataclass(frozen=True)
class IdeaBrief:
    """Everything the per-idea deep dive needs, keyed off a human-readable title."""

    title: str
    candidate_id: str
    rank: int
    elo: int
    category: str
    proposal: str
    description: list[str]
    facts: dict[str, str]
    summary: dict[str, str]
    table_rows: list[tuple[str, str]]
    reviews: list[IdeaReview]
    coherence: list[str]
    deep_verification: list[tuple[str, str]]
    matches: list[IdeaMatch]
    wins: int
    losses: int
    ties: int
    shortlisted: bool
    coherence_notes: list[str] = field(default_factory=list)
    """The standing explanations this idea's coherence paragraphs lean on.

    They are the same sentences under every idea that raises them, so they are
    printed once above the ideas and the paragraphs below carry only the facts.
    """
    support: str = "unknown"
    unresolved_evidence_ids: list[str] = field(default_factory=list)
    accepted_flaw: AdjudicationNote | None = None
    tied_with: int = 0
    """How many other ideas finished on exactly this Elo."""
    deep_verification_lead_in: str = DEEP_VERIFICATION_LEAD_IN
    """What introduces the check list, which depends on where the list came from."""
    contradicting_claims: list[str] = field(default_factory=list)
    """Claims this idea cites that the evidence stage marked as contradicting it.

    A candidate lists the evidence it was built from without saying which way each
    piece cuts, and the report printed the lot as grounding. One live idea cited a
    finding that overly thick coatings degrade ionic conductivity as support for
    applying a coating: the claim is real, the citation is genuine, and it argues
    against the hypothesis carrying it.
    """
    predictions: list[str] = field(default_factory=list)
    """The discriminating predictions, kept unjoined so the grid's copy can be skipped.

    The grid prints the first prediction under "Discriminating prediction" and the
    prose printed the whole list, so every idea in section 4 stated its first
    prediction twice within five lines of itself.
    """
    alternatives: list[str] = field(default_factory=list)
    """The competing readings, kept unjoined because the sentence has to agree with them.

    "The reading it has to displace is that A; and B" reached a live report: one
    reading introducing two, with an "is that" that does not distribute over the
    semicolon. The prose needs the count, and only the list carries it.
    """
    revised_lead_in: str = ""
    """What introduces the rewrite, or empty when evolution did not rewrite this idea."""
    revised_form: list[tuple[str, str]] = field(default_factory=list)
    """The rewrite's fields, restricted to the ones the rewrite changed.

    The report recommended the evolved form of four ideas and printed the evolved text
    of none of them: section 9 named a revision id, listed the changes in the abstract
    ("added sample size and randomization") and left the reader to act on a hypothesis
    the document does not contain. On the live run the recommended form of the leading
    idea was a different coating material at half the loading from the one section 4
    sets out under the same title.
    """
    revised_unchanged: list[str] = field(default_factory=list)
    """Fields the rewrite left alone, named so the diff above can be read as a diff."""
    strategy: str = ""
    """Which of the four generation strategies proposed this idea."""
    mermaid: str = ""
    """The specialist's own workflow diagram, when it drew one."""
    evidence_notes: list[tuple[str, str, str]] = field(default_factory=list)
    """The idea's categorized evidence, as (heading, grounding label, statement).

    A candidate states what it takes to argue for it, against it, and what is
    missing. None of the three reached the page: the report printed the citation
    markers the claim resolved to and dropped the specialist's own reading of
    them, so a reader could see which paper an idea leaned on but not what the
    proposer thought it showed -- nor, anywhere, what the proposer already knew
    was unresolved.
    """
    revised_is_recommended: bool = False
    """Whether the meta-review actually carries the rewrite printed under this idea.

    Evolution rewrites the shortlist, and the meta-review recommends a subset of it or,
    on a run where every candidate carries a fatal flaw, none of it. The section was
    headed "Revised Form Recommended" and opened "This is the form the meta-review
    recommends" either way, so a report whose section 9 says no idea cleared the bar
    told four ideas' readers the opposite in their own sections.
    """

    @property
    def rank_line(self) -> str:
        """The rank, saying so when the number is an ordering the record cannot make.

        Three ideas finished a live tournament on 1184 and were printed as ranks four,
        five and six. Nothing decided that order but the tie-break the sort happened to
        use, and a reader given three consecutive integers has no way to know it.
        """
        if not self.tied_with:
            return f"Rank: {self.rank}"
        # The count is spelled, as every other small count in the report is. Interpolated
        # raw it printed "tied on Elo with 2 other ideas" three rows above "averaging 4.2
        # of five", where the measured mean is the figure and the count is not.
        return (
            f"Rank: {self.rank}, tied on Elo with "
            + (
                "another idea"
                if self.tied_with == 1
                else _plural(self.tied_with, "other idea")
            )
            + " and ordered arbitrarily among them"
        )

    @property
    def governance_notice(self) -> str:
        """The accepted-flaw warning, or empty when nobody overrode a block here.

        For the ranked listing, which the reader meets before the governance block.
        """
        return (
            self.accepted_flaw.notice(adjudications_ahead=True)
            if self.accepted_flaw
            else ""
        )

    @property
    def chapter_governance_notice(self) -> str:
        """The same warning for the idea's own chapter, which follows the block."""
        return (
            self.accepted_flaw.notice(adjudications_ahead=False)
            if self.accepted_flaw
            else ""
        )

    @property
    def support_notice(self) -> str:
        return support_notice(self.support, self.unresolved_evidence_ids)

    @property
    def support_label(self) -> str:
        """The verdict word alone, for where the field's meaning is stated above."""
        return _support_parts(self.support, self.unresolved_evidence_ids)[0]

    @property
    def support_prose(self) -> str:
        """The same verdict without its form label, for use inside a paragraph."""
        return support_prose(self.support, self.unresolved_evidence_ids)

    @property
    def support_is_alarming(self) -> bool:
        """Whether the grounding verdict has to be unmissable rather than noted."""
        return self.support in {"unsupported", "discredited"}

    @property
    def win_rate(self) -> int:
        """Integer percentage, as the reference reports print it."""
        return round(100 * self.wins / len(self.matches)) if self.matches else 0

    def reviews_in(self, section: str) -> list[IdeaReview]:
        return [review for review in self.reviews if review.section == section]


_TITLE_MINOR_WORDS = frozenset(
    """a an the and or nor but for so yet at by from in into of on onto to with without
    as than that which under over via versus vs per is are was were be been
    its their his her our your this these those such it they""".split()
)
# Framing verbs the generator prefixes onto every claim. They describe the act of
# testing rather than the idea, so a title that keeps them all reads the same.
_TITLE_FRAMING_VERBS = frozenset(
    """test tests testing evaluate evaluates assess assesses examine examines determine
    determines investigate investigates compare compares use uses using apply applies
    transfer transfers redesign redesigns design designs develop develops demonstrate
    show shows establish establishes verify verifies explore explores measure measures
    quantify quantifies characterize characterizes implement implements adopt adopts
    combine combines introduce introduces deploy deploys propose proposes""".split()
)
_TITLE_FILLER_WORDS = frozenset("whether that if the a an".split())
_TITLE_CONNECTIVES = re.compile(
    r"\s+(?:because|compared\s+to|compared\s+with|relative\s+to|in\s+order\s+to"
    r"|so\s+that|such\s+that|which|whereas|while|rather\s+than)\s+",
    re.IGNORECASE,
)
_GOAL_RESTATEMENT = re.compile(r"\s*\bfor:\s.*$", re.IGNORECASE | re.DOTALL)


def _title_case(text: str) -> str:
    """Headline case that leaves acronyms and formulae (SPPS, Al2O3) untouched."""
    words = text.split()
    cased = []
    for index, word in enumerate(words):
        if word.strip("().,;:%") in _TITLE_UNITS:
            cased.append(word)
        elif any(character.isupper() for character in word[1:]):
            cased.append(word)
        elif index not in (0, len(words) - 1) and word.lower() in _TITLE_MINOR_WORDS:
            cased.append(word.lower())
        else:
            cased.append(word[:1].upper() + word[1:])
    return " ".join(cased)


# Capitalising a unit changes what it means. Headline case turned every "2 nm"
# coating in the report into "2 Nm", which is a newton-metre, and "1 wt%" into
# "1 Wt%", which is nothing at all. A unit is a symbol, not a word, so it is
# exempt from casing in both directions.
_TITLE_UNITS = frozenset(
    """nm um µm mm cm m nA uA mA A mV V mW W kW mAh Ah mg g kg mol mmol M mM uM
    nS mS S Hz kHz MHz GHz Pa kPa MPa GPa K wt at vol h min s ms us ns rpm ppm
    ppb C F J kJ eV meV mm2 cm2 cm3 mL uL L""".split()
)


def derive_idea_title(claim: str, *, max_words: int = 9) -> str:
    """A short domain title for an idea, because candidate ids are opaque hashes.

    The narrative sections are built around idea names, so every candidate needs one
    even when the generator produced a long templated claim.
    """
    text = _GOAL_RESTATEMENT.sub("", " ".join(claim.split())).strip()
    head = _TITLE_CONNECTIVES.split(text, maxsplit=1)[0]
    # Cutting at a connective only helps when enough of the idea survives it;
    # "Test a boundary condition under which ..." would otherwise become two words.
    if len(head.split()) >= 6:
        text = head
    text = re.split(r"(?<=[a-z0-9])[.;:]\s", text, maxsplit=1)[0]
    words = text.split()
    while words and words[0].strip(",.;:").lower() in _TITLE_FRAMING_VERBS:
        words.pop(0)
        while words and words[0].strip(",.;:").lower() in _TITLE_FILLER_WORDS:
            words.pop(0)
    budget = _title_budget(words, max_words)
    truncated = len(words) > budget
    words = _trimmed_tail(words[:budget], truncated)
    words = _trimmed_tail(_whole_phrase(_closed_brackets(words), truncated), truncated)
    title = " ".join(words).strip(" ,;:.")
    return _title_case(title) if title else "Unnamed Research Idea"


_TITLE_NEGATIONS = frozenset({"not", "never", "no", "cannot", "neither", "nor"})
_TITLE_COORDINATORS = frozenset({"and", "or", "nor"})
_TITLE_BUDGET_SLACK = 5
"""How far past the budget a title may run to finish the phrase it is inside."""


def _title_budget(words: list[str], max_words: int) -> int:
    """How many words this particular claim needs, where the budget cuts it wrong.

    Two cases, both from headings a live run printed. A negated claim trimmed back
    to a whole phrase loses the negation with the words after it: "Ultrathin (1-5
    nm) Metal Oxide Coatings Do Not Fundamentally" was the one idea in the run
    arguing that coatings do not work, and trimming the dangling adverb leaves
    "Ultrathin (1-5 nm) Metal Oxide Coatings" -- the claim with its stance
    removed, under which the section, the summary row and six tournament tables
    all name it. A cut landing immediately before "and" splits a coordination:
    "Applying a 2.5 nm Nanolaminate Coating of Alternating Al2O3" names a
    laminate alternating with nothing, because "and TiO2" was the next two words.

    In both the budget moves rather than the phrase, and never by more than the
    slack -- a title is a title.
    """
    if len(words) <= max_words:
        return max_words
    tokens = [word.strip(",.;:").lower() for word in words]
    if any(token in _TITLE_NEGATIONS for token in tokens[:max_words]):
        # To the end of the clause, which is all that is left of the sentence by
        # this point: a negation reaches its verb or it says the opposite thing.
        if len(words) <= max_words + _TITLE_BUDGET_SLACK:
            return len(words)
    if tokens[max_words] in _TITLE_COORDINATORS:
        for count in range(max_words + 2, max_words + _TITLE_BUDGET_SLACK + 1):
            if count > len(words):
                break
            if not _is_dangling(words[count - 1], True, words[count - 2]):
                return count
    return max_words


def _trimmed_tail(words: list[str], truncated: bool) -> list[str]:
    """Drop trailing tokens until the last one can end a phrase."""
    kept = list(words)
    while kept and _is_dangling(kept[-1], truncated, kept[-2] if len(kept) > 1 else ""):
        kept.pop()
    return kept


def _whole_phrase(words: list[str], truncated: bool) -> list[str]:
    """Drop a closing phrase the cut reduced to a single word.

    "The Primary Mechanism of Cycle Life Extension by Metal" was written about
    metal oxide coatings, and one word after the preposition is what that looks
    like from inside the cut: a modifier standing where its noun should be. Two
    or more words after it and the phrase is doing its own work -- "Coating of
    Al2O3 Nanoparticles" names a thing -- so only the one-word case goes.

    A transitive verb behaves the same way, and worse, because a one-word object
    can look complete: "A 5 nm Boron Nitride (BN) Coating Extends Cycle" was
    written about cycle life, and the title names a coating that extends a cycle.
    """
    if not truncated:
        return words
    for index in range(len(words) - 2, -1, -1):
        head = words[index].strip(",.;:").lower()
        if head in _TITLE_PREPOSITIONS or head in _TITLE_TRANSITIVE_VERBS:
            return words[:index] if index == len(words) - 2 else words
    return words


_TITLE_PREPOSITIONS = frozenset(
    """at by from in into of on onto to with without than via versus per over
    under as""".split()
)


def _closed_brackets(words: list[str]) -> list[str]:
    """Drop a parenthetical the cut left hanging open.

    Claims carry their parameters in brackets, so a nine-word title lands inside
    one often enough to matter: "... Island Coating (5 Nm" and "... Polymer
    Coating (PEDOT:PSS, 20 Nm" were two headings in one report. Reading to the
    close would mean choosing a different length for every idea, so the opening
    bracket and everything after it goes instead.
    """
    depth = 0
    for index, word in enumerate(words):
        depth += word.count("(") - word.count(")")
        if depth > 0 and index == len(words) - 1:
            opened = next(
                position
                for position in range(len(words) - 1, -1, -1)
                if "(" in words[position]
            )
            head = words[opened].split("(", maxsplit=1)[0].strip(" ,;:")
            return words[:opened] + ([head] if head else [])
    return words


# A trailing quantity is always mid-phrase: it was counting something the title lost.
_TITLE_NUMBER_WORDS = frozenset(
    """one two three four five six seven eight nine ten eleven twelve fifteen twenty
    thirty forty fifty sixty seventy eighty ninety hundred thousand half both
    several many few first second third""".split()
)


def _is_dangling(word: str, truncated: bool, previous: str = "") -> bool:
    """Whether a trailing token leaves the title mid-phrase.

    Truncation is what produces the awkward cases: "... Peptide Containing Three"
    only happens because the claim continued, so the extra trims apply there only.
    """
    token = word.strip(",.;:").lower()
    if token in _TITLE_MINOR_WORDS:
        return True
    if not truncated:
        return False
    prior = previous.strip(",.;:").lower()
    # A participle straight after a preposition is describing a noun the cut
    # took away: "Observed in Coated" was "... in coated electrodes". After a
    # noun it is the head of the phrase instead -- "Surface Coating" -- and
    # dropping it costs the title its subject.
    if token.endswith(("ing", "ed")) and prior in _TITLE_MINOR_WORDS:
        return True
    return (
        prior in _TITLE_INSTRUMENT_WORDS
        or token in _TITLE_NUMBER_WORDS
        or token in _TITLE_FRAMING_VERBS
        or token in _TITLE_BOUND_PARTICIPLES
        or token in _TITLE_TRANSITIVE_VERBS
        or token.endswith(_TITLE_ADJECTIVE_SUFFIXES)
        or token.isdigit()
    )


_TITLE_INSTRUMENT_WORDS = frozenset({"via", "using", "through"})
"""Prepositions that open a method name, which is never one word long."""

# Participles that cannot end a phrase because they always take a complement.
# "Observed" stands on its own -- "the extension observed" is a complete thing to
# name -- but "composed", "applied" and "based" do not, and a live run printed
# "An Artificial Cathode-electrolyte Interphase (CEI) Composed" and "A Defect-free
# 2 Nm ZrO2 Coating Applied" as two headings, each stopping one preposition short
# of saying anything.
_TITLE_BOUND_PARTICIPLES = frozenset(
    """composed applied based derived consisting comprising coated made embedded
    doped functionalized functionalised modified loaded grafted anchored bonded
    combined paired compared coupled infused impregnated deposited observed
    reported measured recorded described produced achieved induced driven
    mediated attributed caused triggered sustained""".split()
)
# A transitive verb whose object the cut removed. "... Polyurethane Coating (15 Nm,
# Spray-coated) Extends" names the effect and then loses the thing it acts on.
_TITLE_TRANSITIVE_VERBS = frozenset(
    """extends extend improves improve enhances enhance increases increase reduces
    reduce prevents prevent enables enable suppresses suppress mitigates mitigate
    raises raise lowers lower boosts boost yields yield delivers deliver""".split()
)
# Endings that mark an attributive adjective, which modifies a noun the cut took
# away: "... a 1 Wt% Al2O3 Protective" was a protective coating. Deliberately
# excludes "-al" and "-ic", where the domain's nouns live (metal, material,
# potential, ceramic), and where a wrong drop would cost the title its subject.
_TITLE_ADJECTIVE_SUFFIXES = ("ive", "ous", "able", "ible")


def unique_titles(claims: Sequence[str]) -> list[str]:
    """Title every candidate, disambiguating collisions so cross-references resolve."""
    titles: list[str] = []
    seen: dict[str, int] = {}
    for claim in claims:
        title = derive_idea_title(claim)
        count = seen.get(title.lower(), 0) + 1
        seen[title.lower()] = count
        titles.append(title if count == 1 else f"{title} (Variant {count})")
    return titles


@dataclass(frozen=True)
class ProvenanceNote:
    """How one stage's typed payload was obtained, for the provenance appendix."""

    stage: str
    agent: str
    schema_name: str
    source: str
    repairs: list[str]
    error: str
    model: str = ""
    prompt_version: str = ""
    created_at: str = ""


def _quoted(text: str) -> str:
    """Reprint a person's words unedited, marked as theirs rather than the report's."""
    cleaned = " ".join(str(text or "").split())
    return f'"{cleaned}"' if cleaned else '"(no text was recorded)"'


@dataclass(frozen=True)
class AdjudicationNote:
    """A human's answer to a fatal governance finding, kept verbatim for the reader.

    The flaw and the justification are stored exactly as they were written. A safety
    decision that is summarised cannot be judged by the person reading it, only
    noted, and noting is what the report is supposed to stop being enough.
    """

    candidate_id: str
    title: str
    resolution: str
    adjudicator: str
    justification: str
    fatal_flaws: list[str]
    claim: str = ""

    @property
    def withdrawn(self) -> bool:
        return self.resolution == "withdraw"

    @property
    def heading(self) -> str:
        verb = "Withdrawn" if self.withdrawn else "Override — fatal flaw accepted"
        return f"{verb}: {self.title}"

    @property
    def flaw_text(self) -> str:
        """Every recorded flaw, verbatim, run together only by sentence boundary."""
        flaws = [
            " ".join(str(item).split())
            for item in self.fatal_flaws
            if str(item).strip()
        ]
        return " ".join(flaws) if flaws else "(no flaw text was recorded)"

    @property
    def resolution_sentence(self) -> str:
        if self.withdrawn:
            return (
                f"{self.adjudicator} withdrew this hypothesis from the population in "
                "answer to a fatal flaw recorded by the safety and governance review. "
                "It did not enter the tournament, so it carries no rank and no Elo."
            )
        return (
            f"{self.adjudicator} accepted this fatal flaw and allowed the hypothesis "
            "to stand. The flaw was not fixed, withdrawn or mitigated; the decision "
            # It read "a named person decided the work could proceed", which is a
            # claim about accountability that the name cannot support: it is a
            # free-text argument the run never verifies. What is on the record is
            # the decision and the name given for it, and the governance section
            # says once what that name is worth.
            "on the record is to proceed while carrying it."
        )

    def notice(self, *, adjudications_ahead: bool) -> str:
        """The unmissable paragraph printed wherever this idea appears in the report.

        The flaw travels with the idea and the reason does not. An overridden idea
        carried both into three places -- the listing, its own Critical Flaws
        subsection and the governance block -- so a reader met the same two verbatim
        quotations three times and had to compare them to see they were the same
        decision. What a reader must not be able to miss is the flaw; the reasoning
        that answered it is one thing, set out once, where the decision is.

        The governance block is appended to the research overview, about a sixth of
        the way in. From the ranked listing it is ahead of the reader and from an
        idea's own chapter it is nine hundred lines behind, and the paragraph said
        "below" in both places -- so half the time it sent a reader forward past the
        section it was pointing at. The caller knows which side it is printing on.
        """
        lead = (
            "Withdrawn after review."
            if self.withdrawn
            else "Warning: this idea carries a fatal safety and governance flaw that "
            "was accepted rather than resolved."
        )
        where = "below" if adjudications_ahead else "above"
        return (
            f"{lead} {self.resolution_sentence} The flaw, reprinted verbatim: "
            f"{_quoted(self.flaw_text)} The reason {self.adjudicator} gave for the "
            f"decision is reprinted in full under Governance adjudications {where}."
        )

    @property
    def reprise(self) -> str:
        """The second meeting with the same flaw inside one idea's own chapter.

        The chapter opens with the notice and the Critical Flaws subsection printed
        it again verbatim about sixty lines later, quotation and all, so the reader
        had to compare two long paragraphs to establish they were one decision. The
        subsection is headed Critical Flaws and cannot stay silent about the flaw
        that matters most; it says the flaw is there and where it was just read.
        """
        if self.withdrawn:
            return (
                f"This idea was withdrawn by {self.adjudicator} over the fatal safety "
                "and governance flaw quoted at the head of this section."
            )
        return (
            "The fatal safety and governance flaw quoted at the head of this section "
            f"was accepted by {self.adjudicator} rather than resolved, and it still "
            "stands against this idea."
        )


@dataclass(frozen=True)
class BlockerNote:
    """A fatal governance finding nobody has answered, named by the idea it stops."""

    candidate_id: str
    title: str
    fatal_flaws: list[str]

    @property
    def flaw_text(self) -> str:
        flaws = [
            " ".join(str(item).split())
            for item in self.fatal_flaws
            if str(item).strip()
        ]
        return " ".join(flaws) if flaws else "(no flaw text was recorded)"


@dataclass
class ResearchRecord:
    """Every accepted artifact, typed once, plus the indices the report is built from.

    Each renderer previously re-derived candidate lookups and rank ordering from raw
    payloads; collecting them here keeps the narrative and the deep dives consistent.
    """

    session: Session
    plan: ResearchPlan | None = None
    discovery: DiscoveryManifest | None = None
    evidence: EvidencePacket | None = None
    population: CandidatePopulation | None = None
    reviews: list[ReviewSet] = field(default_factory=list)
    tournament: TournamentState | None = None
    evolution: EvolutionCycle | None = None
    landscape: ResearchLandscape | None = None
    manifest: DossierManifest | None = None
    provenance: list[ProvenanceNote] = field(default_factory=list)
    titles: dict[str, str] = field(default_factory=dict)
    citations: CitationRegistry = field(default_factory=lambda: CitationRegistry([]))
    evidence_support: dict[str, CandidateCitations] = field(default_factory=dict)
    adjudications: list[AdjudicationNote] = field(default_factory=list)
    open_governance_blocks: list[BlockerNote] = field(default_factory=list)
    superseded_populations: int = 0
    cited_evidence: dict[str, list[list[str]]] = field(default_factory=dict)
    """Each candidate's own evidence statements as recorded, before ids were named.

    ``_name_ids_in_prose`` rewrites every id it finds in stored prose into the thing
    it names, and it reaches these statements too. The Evidence Assessment block is
    built afterwards and is the one consumer that needs the ids: it resolves them
    itself, prints what the record holds rather than the id, and reads the badge off
    the record's verification status. Run on the rewritten text it found no ids at
    all, so a live report carried "**[Unsourced claim]** The claim drawn from Ultrathin
    Al2O3 Coatings ..." -- a bullet naming its source and labelled as naming none --
    and, where the specialist had answered with an id and nothing else, the bullet was
    the whole of "**[Unsourced claim]** The unverified cited claim." Keeping the
    original text here lets that block see what it was written to read.
    """

    lineage: dict[str, str] = field(default_factory=dict)
    """Every evolved candidate id, mapped to the ranked ancestor it descends from.

    Evolution mints a new id per revision, and the stages that run after it -- the
    landscape, the meta-review manifest -- refer to whichever id was current when
    they ran. The tournament ranked the originals. Without this map the report
    resolves each id to its own derived title and silently switches vocabulary
    partway through: a live run recommended "Dry-coating NCM811 Cathodes with a 1
    wt% Al2O3", an idea the reader had never been shown, because section four had
    presented its parent as "... with a 2 wt% TiO2".
    """

    @property
    def candidates(self) -> list[Candidate]:
        return list(self.population.candidates) if self.population else []

    @property
    def withdrawals(self) -> list[AdjudicationNote]:
        return [item for item in self.adjudications if item.withdrawn]

    @property
    def overrides(self) -> list[AdjudicationNote]:
        return [item for item in self.adjudications if not item.withdrawn]

    def override_for(self, candidate_id: str) -> AdjudicationNote | None:
        """The accepted fatal flaw a live idea still carries, if a person accepted one."""
        return next(
            (item for item in self.overrides if item.candidate_id == candidate_id),
            None,
        )

    def title_for(self, candidate_id: str) -> str:
        return self.titles.get(candidate_id, "Unnamed Research Idea")

    def ranked_id(self, candidate_id: str) -> str:
        """The id under which this idea was presented, reviewed and ranked."""
        return self.lineage.get(candidate_id, candidate_id)

    def ranked_title(self, candidate_id: str) -> str:
        """The name the reader has already been given for this idea."""
        return self.title_for(self.ranked_id(candidate_id))

    def revisions_of(self, candidate_id: str) -> list[EvolutionRecord]:
        """Every rewrite of this idea, in the order evolution made them.

        The report prints the last one, because that is what would be carried, but the
        change log has to cover the lot: a run that rewrote an idea twice reported only
        the second round, so "revised to version 3: specified H14-grade HEPA filtration"
        stood over a rewrite that had also swapped the coating material and halved the
        loading in round one.
        """
        ranked = self.ranked_id(candidate_id)
        revisions = [
            item
            for item in (self.evolution.records if self.evolution else [])
            if self.ranked_id(item.candidate.id) == ranked
        ]
        return sorted(revisions, key=lambda item: item.candidate.version)

    def revision_of(self, candidate_id: str) -> EvolutionRecord | None:
        """The latest revision of this idea, if evolution produced one."""
        revisions = self.revisions_of(candidate_id)
        return revisions[-1] if revisions else None

    def rereviews_of(self, candidate_id: str) -> list[CandidateReview]:
        """Reviews run against this idea's revisions, after the tournament."""
        ranked = self.ranked_id(candidate_id)
        return [
            review
            for review in (self.evolution.rereviews if self.evolution else [])
            if self.ranked_id(review.candidate_id) == ranked
        ]

    def rereviews_of_latest(self, candidate_id: str) -> list[CandidateReview]:
        """Reviews of the one rewrite the report prints, not of the whole lineage.

        Every round of evolution mints a revision and re-reviews it, and all of them
        resolve back to the same ranked ancestor. The report prints a diff against the
        last revision only, so counting the lineage attributed three rounds of verdicts
        to one rewrite: "it was re-reviewed after the rewrite: 15 reviews said accept"
        over a rewrite five reviewers saw.
        """
        revision = self.revision_of(candidate_id)
        if revision is None:
            return []
        return [
            review
            for review in (self.evolution.rereviews if self.evolution else [])
            if review.candidate_id == revision.candidate.id
        ]

    @property
    def post_evolution_order(self) -> list[str]:
        """Ranked ids in the order the last post-evolution ranking round put them.

        Evolution reruns the tournament on the rewrites, and that round can order them
        differently from the one section 4 sets out. Empty when no such round ran.
        """
        history = self.evolution.ranking_history if self.evolution else []
        if not history:
            return []
        ratings = history[-1].ratings
        ordered = sorted(ratings.items(), key=lambda item: (-item[1], item[0]))
        return list(dict.fromkeys(self.ranked_id(item) for item, _ in ordered))

    @property
    def deep_research_stood_in(self) -> bool:
        """Whether something ran in place of the Deep Research agent.

        Read off the manifest's convergence reason. It lives here rather than beside
        the appendix that first needed it because the narrative body needs it too:
        the thinness of the literature is a fact about which pass ran, and a reader
        who meets the thinness in section three should not have to reach the appendix
        to learn that the stage's designed path never executed.
        """
        return (
            self.discovery is not None
            and self.discovery.convergence_reason in DISCOVERY_STOOD_IN
        )

    def flaw_sections(self, candidate_id: str) -> list[str]:
        """The review sections that recorded a fatal flaw against one idea."""
        ranked = self.ranked_id(candidate_id)
        return sorted(
            {
                CRITERION_SECTIONS.get(review.criterion, "Correctness")
                for review_set in self.reviews
                for review in review_set.reviews
                if review.fatal_flaws and self.ranked_id(review.candidate_id) == ranked
            }
        )

    @property
    def recorded_fatal_flaw_ids(self) -> set[str]:
        """Ideas a reviewer actually recorded a fatal flaw against.

        The meta-review states its own exclusion list, and it is a model's summary of
        the review round rather than a reading of it. On a live run the two disagreed
        in both directions at once: it excluded an idea no reviewer had faulted and
        omitted one carrying a recorded novelty flaw. Printing the summary unchecked
        put a fatal flaw on the record against an idea that never had one.
        """
        return {
            self.ranked_id(review.candidate_id)
            for review_set in self.reviews
            for review in review_set.reviews
            if review.fatal_flaws
        }

    @property
    def judged_by_model(self) -> bool:
        """Whether any match was decided by a judge rather than by arithmetic.

        Where it is false the tournament never read the comparison criteria and the
        ordering is a function of the review scores; where it is true the ordering is a
        function of the matches and the review scores do not enter it. Two sections
        described the run the other way round, so both are asked here.
        """
        comparisons = self.tournament.comparisons if self.tournament else []
        return any(comparison.judge != "deterministic" for comparison in comparisons)

    @property
    def shown_reviews(self) -> list[CandidateReview]:
        """The reviews a reader can find in this report, under the idea they are about.

        A withdrawn idea keeps its reviews in the record and loses its section, so the
        two counts part company: a run that withdrew one of eight ideas told the reader
        it held "40 reviews of the ideas, printed in full under each" above 35 printed
        reviews. Any sentence that promises the reader a place to look has to count
        what is in that place.
        """
        live = {candidate.id for candidate in self.candidates}
        return [
            review
            for review_set in self.reviews
            for review in review_set.reviews
            if self.ranked_id(review.candidate_id) in live
        ]

    def cluster_of(self, candidate_id: str) -> list[str]:
        """The ranked ideas sharing this idea's region, itself included."""
        ranked = self.ranked_id(candidate_id)
        for cluster in self.landscape.clusters if self.landscape else []:
            members = [self.ranked_id(item) for item in cluster.candidate_ids]
            if ranked in members:
                return list(dict.fromkeys(members))
        return [ranked]

    @property
    def fallback_stages(self) -> list[ProvenanceNote]:
        """Stages whose payload is a template rather than the specialist's own work."""
        return [
            note for note in self.provenance if note.source == "deterministic_fallback"
        ]

    @property
    def repaired_stages(self) -> list[ProvenanceNote]:
        return [note for note in self.provenance if note.source == "repaired"]


_SCHEMA_MODELS: dict[str, type[Contract]] = {
    "ResearchPlan": ResearchPlan,
    "DiscoveryManifest": DiscoveryManifest,
    "EvidencePacket": EvidencePacket,
    "CandidatePopulation": CandidatePopulation,
    "ReviewSet": ReviewSet,
    "TournamentState": TournamentState,
    "EvolutionCycle": EvolutionCycle,
    "ResearchLandscape": ResearchLandscape,
    "DossierManifest": DossierManifest,
}


def _typed(payload: dict[str, Any], schema_name: str) -> Any | None:
    """Validate a stored payload, tolerating artifacts written by older schemas."""
    model = _SCHEMA_MODELS.get(schema_name)
    if model is None or not payload:
        return None
    try:
        return model.model_validate(payload)
    except Exception:
        # A report must still render when one stage predates a contract change.
        return None


def _merged_evidence(
    existing: EvidencePacket | None, packet: EvidencePacket
) -> EvidencePacket:
    """Every claim either evidence pass produced, with the later pass's word on it.

    Two stages emit an EvidencePacket: discovery, which extracts the claims, and source
    verification, which is meant to hand them back checked. Assigning the second over
    the first meant a verification pass that came back empty took the discovered claims
    with it -- one live run lost all six, so the report stated the evidence base was
    empty while the ideas below went on citing those claims by id. Keeping both means
    an unreturned claim stays on the page as discovered and unverified, which is what
    it is.
    """
    if existing is None:
        return packet
    claims = {claim.id: claim for claim in existing.claims}
    claims.update({claim.id: claim for claim in packet.claims})
    sources = {source.id: source for source in existing.sources}
    sources.update({source.id: source for source in packet.sources})
    return packet.model_copy(
        update={
            "claims": list(claims.values()),
            "sources": list(sources.values()),
            "limitations": list(
                dict.fromkeys([*existing.limitations, *packet.limitations])
            ),
        }
    )


def _claim_annotations(evidence: EvidencePacket | None) -> dict[str, str]:
    """Derive each source's evidence qualifier from what verification actually found.

    The qualifiers are not editorial: they restate the verification status already
    recorded against the source, so a reader can see which citations are load-bearing.
    """
    annotations: dict[str, str] = {}
    verdicts: dict[str, set[str]] = {}
    if evidence:
        by_id = {source.id: source for source in evidence.sources}
        for claim in evidence.claims:
            source = by_id.get(claim.source_id or "")
            if source is None:
                continue
            # A claim's relation is its bearing on the research question, not a
            # verdict on the source that carries it. Reading "contradicts" as
            # "disputed" tagged a real, uncontested finding -- that overly thick
            # coatings degrade ionic conductivity -- as doubtful, when the only thing
            # true of it is that it argues against the hypothesis. Which way a finding
            # cuts is stated in prose where it is used; it is not a source qualifier.
            if claim.verification_status in {"retracted", "inaccessible"}:
                verdict = "inaccurate"
            elif claim.verification_status in {"verified", "corrected"}:
                verdict = "" if claim.confidence >= 0.8 else "leaning accurate"
            else:
                verdict = "unsupported"
            verdicts.setdefault(source.url, set()).add(verdict)
        for url, found in verdicts.items():
            for candidate in CITATION_ANNOTATIONS:
                if candidate in found:
                    annotations[url] = candidate
                    break
    return annotations


def load_record(session: Session) -> ResearchRecord:
    """Type every accepted artifact once and index the derived report structures."""
    record = ResearchRecord(session=session)
    # A withdrawn hypothesis survives only in the population it was cut from, so the
    # superseded versions are read for their candidates and for nothing else.
    retired: dict[str, Candidate] = {}
    for artifact in session.artifacts:
        if artifact.artifact_type != "specialist_output":
            continue
        if artifact.status == ArtifactStatus.SUPERSEDED:
            superseded = _typed(artifact.payload, artifact.schema_name)
            if isinstance(superseded, CandidatePopulation):
                record.superseded_populations += 1
                for item in superseded.candidates:
                    retired.setdefault(item.id, item)
            continue
        record.provenance.append(
            ProvenanceNote(
                stage=artifact.stage,
                agent=artifact.agent,
                schema_name=artifact.schema_name,
                source=artifact.payload_source,
                repairs=list(artifact.payload_repairs),
                error=artifact.payload_error,
                model=artifact.producer_model,
                prompt_version=artifact.prompt_version,
                created_at=artifact.created_at,
            )
        )
        typed = _typed(artifact.payload, artifact.schema_name)
        if typed is None:
            continue
        if isinstance(typed, ReviewSet):
            record.reviews.append(typed)
        elif isinstance(typed, ResearchPlan):
            record.plan = typed
        elif isinstance(typed, DiscoveryManifest):
            record.discovery = typed
        elif isinstance(typed, EvidencePacket):
            record.evidence = _merged_evidence(record.evidence, typed)
        elif isinstance(typed, CandidatePopulation):
            record.population = typed
        elif isinstance(typed, TournamentState):
            record.tournament = typed
        elif isinstance(typed, EvolutionCycle):
            record.evolution = typed
        elif isinstance(typed, ResearchLandscape):
            record.landscape = typed
        elif isinstance(typed, DossierManifest):
            record.manifest = typed
    # By stage rather than by position in the session. A governance withdrawal writes a
    # replacement population late in the run and appends it, so the table headed "what
    # each stage produced" listed the generate stage after all five reflect rows --
    # telling a reader the hypotheses were written after the reviews of them. The sort
    # is stable, so two artifacts from one stage keep the order they were recorded in.
    record.provenance.sort(key=lambda note: _STAGE_ORDER.get(note.stage, len(STAGES)))
    candidates = record.candidates
    for candidate, title in zip(
        candidates, unique_titles([item.claim for item in candidates]), strict=True
    ):
        record.titles[candidate.id] = title
    if record.evolution:
        for evolved in record.evolution.records:
            record.titles.setdefault(
                evolved.candidate.id, derive_idea_title(evolved.candidate.claim)
            )
        _trace_lineage(record, {item.id for item in candidates})
    leads = list(record.discovery.source_leads) if record.discovery else []
    # The sources the evidence packet carries, where discovery did not lead with them.
    # Only a lead can be numbered, so a claim whose source discovery never listed was
    # printed with no marker and its source stayed out of the reference list -- while
    # ideas cited the claim by id and the report said "every claim this idea cites
    # exists" with nothing a reader could follow to check that.
    known = {lead.canonical_url for lead in leads}
    for source in record.evidence.sources if record.evidence else []:
        if not source.url or source.url in known:
            continue
        known.add(source.url)
        leads.append(
            SourceLead(
                canonical_url=source.url,
                title=source.title,
                source_type=source.source_type,
                provider="source_verification",
            )
        )
    record.citations = CitationRegistry(
        leads, annotations=_claim_annotations(record.evidence)
    )
    if record.population:
        # A candidate may cite an evidence id that names nothing. Resolving here means
        # the report can never present a dangling id as though it were grounding.
        record.evidence_support = resolve_population(record.population, record.evidence)
    _load_governance(record, retired)
    record.cited_evidence = {
        candidate.id: [
            list(candidate.evidence_for),
            list(candidate.evidence_against),
            list(candidate.evidence_gaps),
        ]
        for candidate in record.candidates
    }
    _name_ids_in_prose(record)
    return record


def _named_parents(evolved: EvolutionRecord) -> list[str]:
    """Every parent this revision names, wherever the specialist wrote it down.

    ``EvolutionRecord.parent_ids`` is where the contract puts it, but a specialist
    that returns the flat ``records`` shape routinely puts the parent on the evolved
    candidate instead, and the reshaper that unwraps the by-round shape only reads it
    off the candidate. Either field is the specialist saying the same thing.
    """
    return list(dict.fromkeys([*evolved.parent_ids, *evolved.candidate.parent_ids]))


def _ancestor_in_id(revision_id: str, known: Iterable[str]) -> str:
    """The one ranked id a revision's own id is built out of, if exactly one is.

    Live runs name revisions themselves, and the names they choose carry the parent:
    ``cand_3_evolved_1``, ``evolved_cand_3_v2``. Nothing in the contract requires
    that, so this is a recovery and not the route -- but a run whose lineage does not
    resolve prints a different title for the same idea in the ranking and in the
    recommendation, which is the defect this exists to stop.

    Bounded on both sides so ``cand_1`` is not read out of ``cand_11``, and taken
    only when a single ranked id matches: two matches name no one parent.
    """
    found = [
        item
        for item in known
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(item)}(?![A-Za-z0-9])",
            revision_id,
        )
    ]
    return found[0] if len(found) == 1 else ""


def _trace_lineage(record: ResearchRecord, ranked: set[str]) -> None:
    """Walk each revision back to the candidate the tournament actually ranked.

    A second revision names its first revision as parent, not the original, so one
    pass over the records resolves a v2 and leaves a v3 dangling whenever the
    specialist emitted them out of order. Repeating until nothing new resolves costs
    a handful of passes over a list that never exceeds the population size.
    """
    # A parent named by title rather than by id. The reader-facing title is what the
    # specialist has in front of it, and naming ideas by it is the commonest way the
    # id goes missing. Exact after whitespace, and dropped where two ideas share it.
    by_title = {
        " ".join(title.split()).casefold(): candidate_id
        for candidate_id, title in record.titles.items()
        if candidate_id in ranked
    }
    for title, count in Counter(
        " ".join(record.titles[item].split()).casefold()
        for item in ranked
        if item in record.titles
    ).items():
        if count > 1:
            by_title.pop(title, None)
    pending = list(record.evolution.records if record.evolution else [])
    while pending:
        unresolved = []
        for evolved in pending:
            named = _named_parents(evolved)
            ancestor = next(
                (
                    parent if parent in ranked else record.lineage[parent]
                    for parent in named
                    if parent in ranked or parent in record.lineage
                ),
                "",
            ) or next(
                (
                    by_title[key]
                    for parent in named
                    if (key := " ".join(parent.split()).casefold()) in by_title
                ),
                "",
            )
            if ancestor:
                record.lineage[evolved.candidate.id] = ancestor
            else:
                unresolved.append(evolved)
        if len(unresolved) == len(pending):
            # Nothing left resolves through what the records name. Before giving up,
            # read the ancestor out of the revision's own id, which is where a live
            # specialist that omitted parent_ids altogether still put it.
            still_unresolved = []
            for evolved in unresolved:
                found = _ancestor_in_id(
                    evolved.candidate.id, [*ranked, *record.lineage]
                )
                if found:
                    record.lineage[evolved.candidate.id] = record.lineage.get(
                        found, found
                    )
                else:
                    still_unresolved.append(evolved)
            if len(still_unresolved) == len(unresolved):
                # Every remaining revision descends from something outside this
                # population. They are left unmapped rather than guessed at.
                break
            unresolved = still_unresolved
        pending = unresolved


def _load_governance(record: ResearchRecord, retired: dict[str, Candidate]) -> None:
    """Index the human answers to fatal governance findings, and the unanswered ones.

    A hypothesis that was withdrawn is no longer in the population, so it would
    otherwise vanish from a numbered list without explanation. Recovering it here is
    what lets the report show a gap instead of quietly renumbering around it.
    """
    session = record.session
    for adjudication in session.governance_adjudications:
        candidate = retired.get(adjudication.candidate_id)
        title = record.titles.get(adjudication.candidate_id)
        if title is None:
            title = (
                derive_idea_title(candidate.claim)
                if candidate
                else "Unnamed Withdrawn Idea"
            )
            record.titles[adjudication.candidate_id] = title
        record.adjudications.append(
            AdjudicationNote(
                candidate_id=adjudication.candidate_id,
                title=title,
                resolution=adjudication.resolution,
                adjudicator=adjudication.adjudicator,
                justification=adjudication.justification,
                fatal_flaws=list(adjudication.fatal_flaws),
                claim=_sentence(candidate.claim) if candidate else "",
            )
        )
    record.open_governance_blocks = [
        BlockerNote(
            candidate_id=blocker.candidate_id,
            title=record.title_for(blocker.candidate_id),
            fatal_flaws=list(blocker.review.fatal_flaws),
        )
        for blocker in open_blockers(session)
    ]


# Specialists quote record ids inside their own prose ("relies entirely on claim_001").
# The id means nothing to a reader, and one that resolves to nothing reads as grounding
# the report cannot produce, so each is replaced by what it actually refers to.
# "src" is here because the evidence stage numbers its sources that way and a live
# run printed "particularly src_1 regarding whether pinhole defects ..." into the
# open questions, where a reader has nothing to resolve it against.
#
# Every part of the id, not just the first: the evidence stage numbers a claim
# "claim_11_1" and the discovery passes number a statement "stmt_5_pass4", and a
# pattern that stopped at the first underscore-separated part matched neither. A
# live transcript therefore read "relies on verified evidence (claim_11_1,
# source_11_2)", and a live idea cited "(lead_0f651732f8364b01)".
_RECORD_ID = re.compile(
    # A discovery statement is filed under the pass that found it, "pass4_stmt_5",
    # and an underscore is a word character: there is no boundary in front of the
    # "stmt", so the pattern never saw one of these. Chapters five and six of a live
    # report printed them raw, in the middle of a reviewer's sentence, while every
    # other chapter named the finding.
    r"\b(?:pass[0-9]*_)?"
    r"(?:claim|source|src|candidate|cand|review|rev|hypothesis|lead|stmt|statement)"
    r"[0-9]*(?:_[0-9a-zA-Z]+)+\b",
    # A reviewer that opens a sentence with an id capitalises it, and "Claim_1 and
    # the source ... already explore dry-coating methods" reached a live report.
    re.IGNORECASE,
)


def _standing(status: str) -> str:
    """The word that has to travel with a record's name wherever prose cites it.

    A debate panelist wrote "this idea relies on verified evidence (claim_11_1,
    source_11_2)" about two records the run had never retrieved, and the report
    printed it under the idea as an argument for ranking it above another. The
    badges and the support verdict say what the grounding is worth elsewhere in
    the document; inside a sentence asserting the opposite they are too far away
    to be read as a correction, so the standing goes into the name itself.
    """
    if status in GROUNDED_STATUSES:
        return ""
    if status == "retracted":
        return "retracted "
    if status == "inaccessible":
        return "unretrieved "
    return "unverified "


# A claim id and the id of the source it was drawn from often appear side by side in
# the same sentence, because the specialist that wrote it was citing both. Naming each
# of them printed "the claim drawn from Dry-coating of NCM811 and the source
# Dry-coating of NCM811", which reads as two sources where the record holds one.
# "The" is captured rather than matched so the rewrite can give it back: the id it
# replaced may have opened the sentence, and a literal lowercase replacement put a
# small "the" at the head of a paragraph.
_DOUBLE_NAMED_SOURCE = re.compile(
    r"(the) ((?:retracted |unretrieved |unverified )?)claim drawn from (.+?) "
    r"and the (?:retracted |unretrieved |unverified )?source \3\b",
    re.IGNORECASE,
)


def _double_named(match: re.Match[str]) -> str:
    """Name the pair once. Two subjects were named, so two have to survive.

    Collapsing them to one left the reviewer's plural verb stranded -- "the claim
    drawn from X already explore dry-coating methods" -- and naming both printed the
    same title twice in one clause.
    """
    return (
        f"{match.group(1)} {match.group(2)}claim drawn from {match.group(3)} "
        "and that source"
    )


# The generator names its own strategy in prose, and the debate agents quote that name
# back verbatim. It is a snake_case enum, so "its generation strategy
# (\"competing_explanation\")" reached a transcript in a live report.
_STRATEGY_PROSE = {
    "evidence_first": "evidence-first",
    "mechanism_first": "mechanism-first",
    "analogy_transfer": "analogy transfer",
    "competing_explanation": "competing explanation",
    "insufficient_evidence": "insufficient evidence",
}
_STRATEGY_ENUM = re.compile("|".join(sorted(_STRATEGY_PROSE, key=len, reverse=True)))

# Vocabulary the specialists share with the pipeline but not with the reader. Each of
# these reached a live report inside a reviewer's own sentence. "Candidate proposes
# that ZnO acts as a chemical HF scavenger" names a row in the population table, and
# a reader who has been reading about ideas for forty pages meets a new noun for the
# thing they have been reading about. "No evidence available in the provided packet"
# names the JSON the reviewer was handed; the reader was handed a report.
_HOUSE_TERMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:provided|current|given|supplied|attached|evidence)\s+packet\b",
            re.IGNORECASE,
        ),
        "session record",
    ),
    (re.compile(r"\bpackets?\b", re.IGNORECASE), "session record"),
    # Only where it is unmistakably the subject: a capital followed by a lower-case
    # word is a sentence opening on the noun, never "Candidate Ideas" or a title.
    (re.compile(r"\bCandidate\b(?= [a-z])"), "The idea"),
    (re.compile(r"\b(the|this|that|The|This|That) candidate\b"), r"\1 idea"),
    # "reflection" is the pipeline's own name for the evidence-and-correctness pass.
    # It is stripped from the review headings, so a debate turn that said "the
    # reflection review points out" left the report calling one review two different
    # things, one of them a stage id.
    (
        re.compile(r"\breflection review\b", re.IGNORECASE),
        "evidence and correctness review",
    ),
)


def _record_names(record: ResearchRecord) -> dict[str, str]:
    """Every id this run holds, mapped to the phrase that stands in for it in prose."""
    names: dict[str, str] = dict(record.titles)
    if record.evidence:
        # The title as the search handed it over ends in the site it was found on
        # and the label on the link. Substituted into a reviewer's sentence, "the
        # source Corrosion-inspired design of artificial interphases ... - DOI
        # doi.org" and "the source One Unrecorded Polymer Batch Number Skewed a
        # Battery Cycling Study mergerotgames.com" both reached a live report,
        # reading as though the furniture were the last words of the paper's name.
        titles = {
            source.id: _without_search_chrome(" ".join(source.title.split()))
            for source in record.evidence.sources
        }
        for source in record.evidence.sources:
            title = titles[source.id]
            standing = _standing(source.verification_status)
            names[source.id] = (
                f"the {standing}source {title}" if title else f"the {standing}source"
            )
        for claim in record.evidence.claims:
            title = titles.get(claim.source_id or "", "")
            standing = _standing(claim.verification_status)
            names[claim.id] = (
                f"the {standing}claim drawn from {title}"
                if title
                # A claim whose source went unnamed has nothing to be called after
                # except what it says. Called after its standing alone, three
                # different records reached a live review inside one parenthesis as
                # "..., the unverified cited claim, the unverified cited claim" --
                # the same four words twice over, naming neither which claims were
                # meant nor what either of them held.
                else _named_by_text(
                    f"the {standing}claim that",
                    claim.claim,
                    f"the {standing}cited claim",
                )
            )
    for lead in record.discovery.source_leads if record.discovery else []:
        title = _without_search_chrome(" ".join(lead.title.split()))
        standing = _standing(lead.verification_status)
        names.setdefault(
            lead.id,
            f"the {standing}source {title}" if title else f"the {standing}source lead",
        )
    for narrative in record.discovery.narratives if record.discovery else []:
        for statement in narrative.statements:
            names.setdefault(
                statement.id,
                _named_by_text(
                    "the finding that",
                    statement.text,
                    "an unverified finding from the literature search",
                ),
            )
    return names


def _named_by_text(opener: str, text: str, fallback: str) -> str:
    """A record called after what it says, where what it says will fit in a sentence.

    A finding is a sentence, and a sentence spliced into the middle of a reviewer's
    own sentence is only readable while it is short. Past that the report says what
    kind of record was cited and leaves the finding where it is printed in full.
    """
    spoken = " ".join(text.split()).rstrip(".")
    if not 0 < len(spoken) <= 120:
        return fallback
    return f"{opener} {spoken[:1].lower()}{spoken[1:]}"


def _name_ids_in_prose(record: ResearchRecord) -> None:
    """Rewrite internal ids into the things they name, everywhere prose is stored."""
    names = _record_names(record)

    # The ids are matched case-insensitively because a reviewer that opens a sentence
    # with one capitalises it, so the lookup has to be case-insensitive too. It was
    # not: "Claim_1 and the source ... already explore dry-coating methods" missed the
    # `claim_1` the record holds and printed "an evidence id that does not exist in
    # this session" as the subject of the sentence -- the renderer reporting its own
    # failed lookup, in prose, in the middle of a review.
    folded = {key.lower(): value for key, value in names.items()}

    def _named(match: re.Match[str]) -> str:
        # An id this run cannot place is set as the identifier it is, the way the
        # integrity lines set theirs. Described instead, it collapsed: a live review
        # read "cites several invalid evidence IDs (a record this session does not
        # hold, a record this session does not hold, a record this session does not
        # hold)", where the reader cannot tell how many distinct ids that is, whether
        # any two are the same, or which one to go and look for.
        if match.group(0).lower() not in folded:
            return f"`{match.group(0)}`"
        name = folded[match.group(0).lower()]
        # An id that opened the sentence takes the sentence's capital with it.
        opens = match.start() == 0 or match.string[: match.start()].rstrip().endswith(
            (".", "!", "?", ":")
        )
        return name[:1].upper() + name[1:] if opens else name

    def replace(text: str) -> str:
        if _RECORD_ID.search(text):
            # A trailing quote around the id would be orphaned by the substitution.
            text = (
                _RECORD_ID.sub(_named, text).replace("'the ", "the ").replace("' ", " ")
            )
            text = _DOUBLE_NAMED_SOURCE.sub(_double_named, text)
        # A field holding nothing but the enum is the enum, not prose about it, and
        # the renderer formats those itself -- rewriting them here turned every
        # "Evidence First" category path into "Evidence-First".
        if text.strip() in _STRATEGY_PROSE:
            return text
        text = _STRATEGY_ENUM.sub(lambda match: _STRATEGY_PROSE[match.group(0)], text)
        for pattern, replacement in _HOUSE_TERMS:
            text = pattern.sub(replacement, text)
        return text

    for model in (
        record.population,
        record.tournament,
        record.evolution,
        record.landscape,
        record.discovery,
        # The manifest writes the shortlist's open questions, and it quotes the
        # sources it wants read in full by id like every other specialist does.
        record.manifest,
        *record.reviews,
    ):
        if model is not None:
            _scrub_prose(model, replace)


def _scrub_prose(value: Any, replace: Callable[[str], str]) -> Any:
    """Apply a text rewrite to every prose field of a contract, sparing id fields."""
    if isinstance(value, str):
        return replace(value)
    if isinstance(value, list):
        return [_scrub_prose(item, replace) for item in value]
    if isinstance(value, Contract):
        for name in type(value).model_fields:
            if name == "id" or name.endswith(("_id", "_ids")):
                continue
            setattr(value, name, _scrub_prose(getattr(value, name), replace))
        return value
    return value


# Label and body separately. The label heads the notice where it stands as its own
# block, which is what it is for; folded into a running paragraph -- "It finished rank
# 1 on an Elo of 1290. Evidence support: uncited. This idea cites no evidence..." --
# a form field lands between two ordinary sentences and reads as a rendering fault.
_SUPPORT_NOTICES = {
    "grounded": (
        "grounded",
        "Every claim this idea cites exists in this report and has been verified "
        "against its source.",
    ),
    "partially_grounded": (
        "partially grounded",
        "Some of the claims this idea cites are verified and the rest are discovered "
        "leads that nobody has confirmed, so the idea is only as strong as the "
        "verified part of its grounding.",
    ),
    "unverified": (
        "unverified",
        "Every claim this idea cites exists, but none of them has been verified "
        "against its source, so the idea rests on retrieved text rather than on "
        "checked evidence.",
    ),
    # "which is an honest position rather than a defect" used to close this notice.
    # It is the report exonerating what its own reviewers penalised: on a live run
    # the evidence-correctness reviews marked these ideas down for exactly the
    # absence the notice was calling honest. The absence is a fact; whether it is a
    # defect is the reviewers' call and they have already made it.
    "uncited": (
        "uncited",
        "This idea cites no evidence anywhere in this report. Nothing here says "
        "whether evidence for it exists — only that none was retrieved and none was "
        "checked — so it stands on its reasoning, and the reviews below are the only "
        "assessment of it.",
    ),
    "unknown": (
        "not resolved",
        "This report was compiled without citation resolution, so nothing here "
        "confirms that the idea's grounding exists.",
    ),
}


def _support_parts(support: str, unresolved: Sequence[str]) -> tuple[str, str]:
    """The grounding verdict as a label and a body, worst cases stated first."""
    if support == "unsupported":
        # The one place in the body an internal id is printed, because a warning that
        # will not say which citation broke cannot be acted on. The ids are bare noun
        # phrases and _join punctuates a series of stated clauses, so a pair of them
        # came out as "claim_1_2, and stmt_3_pass2" -- pointed like a list with an
        # item missing between them. Set in code font besides, so a reader can see
        # that what they are being shown is a literal identifier and not a mangled
        # word.
        named = (
            _names([f"`{item}`" for item in unresolved if str(item).strip()]).strip()
            or "an id it did not record"
        )
        return (
            "",
            "Warning: this idea cites evidence that does not exist anywhere in this "
            f"report — {named}. Nothing grounds the claim below. The citation was "
            "written by the generator and never resolved to a record, so the idea "
            "must be read as unsupported rather than as evidence-backed.",
        )
    if support == "discredited":
        return (
            "",
            "Warning: this idea cites evidence that was retracted or could not be "
            "retrieved. Its stated grounding is discredited, and the claim below "
            "should not be carried forward until it is re-grounded on evidence that "
            "still stands.",
        )
    return _SUPPORT_NOTICES.get(support, _SUPPORT_NOTICES["unknown"])


def support_notice(support: str, unresolved: Sequence[str]) -> str:
    """The verdict as its own block, under the label that says what it is."""
    label, body = _support_parts(support, unresolved)
    return f"Evidence support: {label}. {body}" if label else body


def support_prose(support: str, unresolved: Sequence[str]) -> str:
    """The same verdict as a sentence, for where it is folded into a paragraph."""
    return _support_parts(support, unresolved)[1]


# The same notices written of the field rather than of one idea. A run whose evidence
# stage verified nothing gives every idea the same verdict, and the notice explaining
# what that verdict means was then printed in full under all eight of them.
_SHARED_SUPPORT_BODIES = {
    "grounded": "Every claim any of them cites exists in this report and has been "
    "verified against its source.",
    "partially_grounded": "Each cites some claims that were verified and some that are "
    "discovered leads nobody has confirmed, so each is only as strong as the verified "
    "part of its grounding.",
    "unverified": "Every claim they cite exists, but none has been verified against "
    "its source, so each idea rests on retrieved text rather than on checked evidence.",
    "uncited": "They cite no evidence anywhere in this report. Nothing here says "
    "whether evidence for them exists — only that none was retrieved and none was "
    "checked — so each stands on its reasoning, and the reviews under it are the only "
    "assessment of it.",
    "unknown": "This report was compiled without citation resolution, so nothing here "
    "confirms that their grounding exists.",
}


def shared_support_notices(
    supports: Sequence[str], *, detail: bool = True
) -> tuple[str, set[str]]:
    """What the recurring grounding verdicts mean, said once, and which those are.

    A verdict that only one idea carries stays under that idea: hoisting it would cost
    the reader a page-turn and save nothing. What was worth hoisting on a live run were
    the two that eight ideas shared between them, printed in full eight times.

    The two alarming verdicts have no entry in the shared bodies and so are never
    hoisted, however many ideas carry them: a reader must meet those under the idea.

    The verdicts are reported twice in the report -- beside each idea where the ideas
    are listed, and again where each is examined in detail -- so what they mean is
    explained at the first of those (``detail``) and pointed at from the second.
    """
    counts = Counter(item for item in supports if item in _SHARED_SUPPORT_BODIES)
    recurring = sorted(
        (item for item, count in counts.items() if count > 1),
        key=lambda item: (-counts[item], item),
    )
    if not recurring:
        return "", set()
    if not detail:
        return (
            "Each idea below carries a grounding verdict under its title: "
            + _names(
                [
                    f"{_number_word(counts[item]).lower()} are marked "
                    f"{_SUPPORT_NOTICES[item][0]}"
                    for item in recurring
                ]
            )
            + ". What those verdicts mean is set out under Candidate Ideas above, "
            "where the ideas are first listed.",
            set(recurring),
        )
    parts = [
        f"{_number_word(counts[item])} of them are marked "
        f"{_SUPPORT_NOTICES[item][0]}. {_SHARED_SUPPORT_BODIES[item]}"
        for item in recurring
    ]
    return (
        "Each idea below carries a grounding verdict. What the recurring ones mean is "
        "the same wherever they appear, so it is set out here rather than under every "
        "idea that has one. " + " ".join(parts),
        set(recurring),
    )


def shared_review_questions(
    briefs: Sequence[IdeaBrief],
) -> tuple[list[str], set[tuple[str, str, str]]]:
    """Who each review is and what it asked, said once, and which pairs those are.

    A review's reviewer and its question are properties of the review, not of the idea
    under it: the correctness review asks the same sentence of the first idea and the
    eighth. Printed in place that was five headings, five role names and five
    questions repeated under every idea -- forty fixed paragraphs on a live run of
    eight, between the reader and the findings that actually differ. A pair only one
    idea carries is left where it is; hoisting it would cost a page-turn and save
    nothing.
    """
    counts = Counter(
        (review.section, review.lead_in.rstrip(":"), review.question)
        for brief in briefs
        for review in brief.reviews
    )
    ordered = {name: index for index, name in enumerate(REVIEW_SECTIONS)}
    shared = sorted(
        (item for item, count in counts.items() if count > 1),
        key=lambda item: (ordered.get(item[0], len(ordered)), item[1]),
    )
    if not shared:
        return [], set()
    return (
        [
            "Every idea below is reviewed under the same headings, and each review "
            "asks the same question of every idea it reviews. Who asks what is set "
            "out here rather than under each idea; what varies, and what is printed "
            "there, is the answer.",
            "",
            *[
                f"- **{section}** — {reviewer}. {question}"
                for section, reviewer, question in shared
            ],
            "",
        ],
        set(shared),
    )


def shared_coherence_notes(briefs: Sequence[IdeaBrief]) -> list[str]:
    """The standing explanations the coherence paragraphs lean on, in a fixed order.

    Unlike the grounding verdicts, these are hoisted even when a single idea raises
    one: they define what a spread of scores means rather than describe an idea, and
    a definition belongs with the other definitions above the ideas.
    """
    raised = {note for brief in briefs for note in brief.coherence_notes}
    notes = [note for note in _COHERENCE_NOTES if note in raised]
    if not notes:
        return []
    return [
        "Each idea below closes its reviews on whether they agree with each other. "
        "What a disagreement between them means, and what would settle one, is the "
        "same for every idea and is set out here; under each idea is that idea's own "
        "spread and the review at the bottom of it.",
        "",
        *[line for note in notes for line in (note, "")],
    ]


INTEGRITY_CASES = {
    "unresolved": "its evidence is absent from this session",
    "discredited": "its evidence was retracted or could not be retrieved",
    "unverified": "its evidence was never checked against its source",
    "uncited": "it cites no evidence at all",
}
"""The four ways a grounding can fail, in the order the lines below print them.

The appendix lead-in named all four every time and the list under it never held more
than two, so a reader counting cases against lines came up short. It now names the
ones this run produced, which is what this mapping is separated out for.
"""


def evidence_integrity_lines(record: ResearchRecord) -> list[str]:
    """One line per idea whose stated grounding does not hold, named by its title."""
    return [line for _case, line in _integrity_entries(record)]


def evidence_integrity_cases(record: ResearchRecord) -> list[str]:
    """Which of the four failures this run actually recorded, in printing order."""
    ordered = dict.fromkeys(case for case, _line in _integrity_entries(record))
    return [INTEGRITY_CASES[case] for case in ordered]


def _integrity_entries(record: ResearchRecord) -> list[tuple[str, str]]:
    """Each integrity line paired with the case it states, so both can be named.

    ``citations.integrity_warnings`` names candidates by id; this report has titles
    for them, and an id a reader cannot place is a warning they cannot act on.

    The order is the tournament's, which is the order every other list of ideas in the
    report uses. Sorting on the id sorted on a string the reader never sees, so the
    same five ideas appeared in one order here and another everywhere else, and the
    idea a reader would act on first was fourth in the list of ideas whose grounding
    does not hold.
    """
    ratings = record.tournament.ratings if record.tournament else {}
    ordered = sorted(
        record.evidence_support.items(),
        key=lambda item: (-ratings.get(record.ranked_id(item[0]), 0.0), item[0]),
    )
    # A discredited citation names a record this run does hold, and which paper was
    # withdrawn is the whole of what a reader can act on here. The line printed the
    # ids instead -- "cites evidence that was retracted or could not be retrieved:
    # claim_6_1, source_6_2" -- so the one section of the report about evidence that
    # cannot be trusted was the one place that never said what the evidence was.
    names = _record_names(record)
    lines: list[tuple[str, str]] = []
    grouped: dict[str, list[str]] = {"unverified": [], "uncited": []}
    for candidate_id, citations in ordered:
        title = record.title_for(candidate_id)
        if citations.unresolved:
            lines.append(
                (
                    "unresolved",
                    f"{title} cites evidence that does not exist in this session — "
                    # Nothing to name these after, so they are set as the literal
                    # identifiers they are rather than dressed up as prose.
                    + _names([f"`{item}`" for item in citations.unresolved])
                    + ". Its claim is unsupported.",
                )
            )
        elif citations.discredited:
            lines.append(
                (
                    "discredited",
                    f"{title} cites evidence this session could not stand behind: "
                    + _names(
                        [names.get(item, f"`{item}`") for item in citations.discredited]
                    )
                    + ". Its claim is discredited.",
                )
            )
        # Citing nothing was treated as nothing to report, so the one idea in the run
        # with no grounding whatsoever was the one idea missing from the list of ideas
        # whose grounding does not hold -- and on a live run that idea finished first.
        elif citations.support in grouped:
            grouped[citations.support].append(title)
    # These two cases carry nothing that varies between the ideas in them: no ids to
    # name and no differing reason. One line each printed the same sentence five times
    # over and then the same other sentence three times. The case is stated once and
    # the ideas it covers are named after it.
    hypothesis = grouped["unverified"]
    if hypothesis:
        lines.append(
            (
                "unverified",
                f"{_opening(len(hypothesis), 'idea')} "
                + ("rests" if len(hypothesis) == 1 else "rest")
                + " on evidence that was discovered but never verified, so "
                + (
                    "its claim is"
                    if len(hypothesis) == 1
                    else "each of their claims is"
                )
                + f" a hypothesis: {_joined_titles(hypothesis)}.",
            )
        )
    conjecture = grouped["uncited"]
    if conjecture:
        lines.append(
            (
                "uncited",
                f"{_opening(len(conjecture), 'idea')} cite"
                + ("s" if len(conjecture) == 1 else "")
                + " no evidence at all. The "
                + (
                    "specialist that proposed it"
                    if len(conjecture) == 1
                    else "specialists that proposed them"
                )
                + " recorded no source, so nothing in this session grounds "
                + ("it" if len(conjecture) == 1 else "them")
                + " either way and "
                + ("its claim is" if len(conjecture) == 1 else "each claim is")
                + f" a conjecture: {_joined_titles(conjecture)}.",
            )
        )
    return lines


_STUB_VALUES = frozenset({"n/a", "na", "none", "unknown", "-"})


def _comparable(text: str) -> str:
    """One statement reduced to what makes two of them the same statement."""
    return " ".join(str(text or "").split()).casefold().rstrip(".?!")


def _table_cells(line: str) -> list[str]:
    """The cells of a Markdown table row, or nothing if the line is not one."""
    stripped = line.strip()
    if not stripped.startswith("|") or stripped.count("|") < 2:
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


_RULE_CELL = re.compile(r":?-{2,}:?")


def _row_as_prose(header: Sequence[str], row: Sequence[str]) -> str:
    """One table row as a clause: the first column names it, the rest describe it."""
    pairs = [
        (header[index] if index < len(header) else "", row[index])
        for index in range(1, len(row))
        if row[index]
    ]
    if not pairs:
        return row[0]
    if len(pairs) == 1:
        return f"{row[0]}: {pairs[0][1].rstrip('.')}"
    body = "; ".join(
        f"{name}: {value.rstrip('.')}" if name else value for name, value in pairs
    )
    return f"{row[0]} ({body})"


def _blocks_as_prose(text: str) -> str:
    """A specialist's own Markdown, written as the prose the field is declared to hold.

    ``rationale`` is a prose field, and evolution's rewrite instruction asks for "a
    structured Evaluation Table in the mechanism_model". Four of today's eight ideas
    came back with a Markdown table inside a prose field, and the report printed it
    into a cell of its own Markdown table -- where every pipe the specialist wrote
    ended the cell, so a two-column grid grew to eleven columns the header has no
    names for and the row below it stopped lining up. The same string was printed
    into a paragraph as one line of ``| Category | Description | Judgment |``.

    Nothing is dropped: each row becomes a clause naming what the row was about.
    """
    lines = str(text or "").splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        header = _table_cells(lines[index])
        rule = _table_cells(lines[index + 1]) if index + 1 < len(lines) else []
        if not (
            header
            and rule
            and len(rule) == len(header)
            and all(_RULE_CELL.fullmatch(cell) for cell in rule)
        ):
            heading = _MARKDOWN_HEADING_RE.match(lines[index].strip())
            out.append(
                f"{heading.group(2).strip().rstrip('.')}." if heading else lines[index]
            )
            index += 1
            continue
        index += 2
        rows = []
        while index < len(lines) and (row := _table_cells(lines[index])):
            rows.append(row)
            index += 1
        out.append(" ".join(f"{_row_as_prose(header, row)}." for row in rows))
    return "\n".join(out)


def _sentence(text: str, *, fallback: str = "Not stated by the specialist.") -> str:
    """Normalise a payload string into one sentence, never an empty or 'N/A' stub.

    A field that arrives holding a serialised payload rather than prose is discarded
    instead of printed: a reader cannot audit a JSON blob, and a report that shows one
    has stopped being a report. The fallback says so rather than hiding the gap.
    """
    cleaned = " ".join(_blocks_as_prose(text).split())
    if not cleaned or cleaned.lower() in _STUB_VALUES:
        return fallback
    if _looks_serialised(cleaned):
        return fallback
    if cleaned[-1] not in ".!?":
        cleaned += "."
    # Counts are spelled out in prose, and a spelled count is lower case wherever it is
    # not opening a sentence -- which is decided here, not where the count was built.
    # "one hypothesis was withdrawn from the population and did not compete." reached a
    # live report as its own sentence. Only the number words are touched: a payload
    # string may legitimately open on pH, n-hexane or a lower-case gene symbol, and
    # capitalising those would be a worse error than the one being fixed.
    first = cleaned.partition(" ")[0]
    if first in _LOWER_NUMBER_WORDS:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _looks_serialised(text: str) -> bool:
    """Whether a field holds a serialised structure instead of the prose it promised."""
    return "{" in text or "}" in text or '": ' in text


def _spliced(text: str) -> str:
    """Fold a stated sentence into a longer one without destroying its notation.

    Only the opening word is lowercased, and only when it is ordinary prose. Applying
    ``str.lower`` to the whole sentence turns LiPF6 into lipf6 and ICP-MS into icp-ms,
    which a reader takes for a transcription error rather than for a sentence.

    Sparing all-capitals was not enough to spare the formulae: a mixed-case name is
    neither upper nor lower, so "ZnO dissolves into the electrolyte" was folded into
    a sentence as "zno dissolves", and "LiF coating increases interfacial impedance"
    as "lif coating". The test is instead whether the word is merely capitalised --
    one leading capital over an otherwise lower-case word, which is what a sentence
    does to a word that would not otherwise carry one.
    """
    cleaned = " ".join(str(text or "").split()).rstrip(".")
    if not cleaned:
        return cleaned
    head, separator, tail = cleaned.partition(" ")
    if (
        head[:1].isupper()
        and (head[1:].islower() or not head[1:])
        and head.strip(",;:").lower() not in _EPONYMS
    ):
        head = head.lower()
    return head + separator + tail


# A surname a method is named after is a merely-capitalised word by the test above, so
# "Karl Fischer titration to ensure moisture content ... is below 50 ppm" was folded
# into a list of go/no-go tests as "karl Fischer titration". Nothing in the shape of
# the sentence separates that from "Initial Coulombic efficiency will be improved",
# which is a word the sentence capitalised and has to fold: both are a capitalised
# word followed by another. So the eponyms are named. A name that is not on this list
# folds down, which is what happened before the list existed -- it fixes the ones this
# corpus actually uses rather than pretending to a rule it cannot have.
_EPONYMS = frozenset(
    {
        "arrhenius",
        "bayesian",
        "brunauer",
        "fourier",
        "gaussian",
        "karl",
        "nyquist",
        "raman",
        "rietveld",
        "tafel",
        "warburg",
    }
)


# A full stop that ends a sentence rather than an abbreviation. The corpus is full of
# "e.g. ZnO" and "Fig. 3", which look exactly like a sentence boundary to a pattern that
# only checks for a stop followed by a capital.
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "cf",
        "dr",
        "e.g",
        "eq",
        "et al",
        "fig",
        "i.e",
        "no",
        "ref",
        "vs",
    }
)
_SENTENCE_BREAK = re.compile(r"(\S*)([.!?][\"')\]]*)\s+(?=[A-Z])")


# Heads that cannot carry a conjunction in front of them. A subordinator opens a
# clause that needs a main clause after it, so "; and while known, quantifying the
# crossover point could help" hangs a subordinate clause off a conjunction with
# nothing for it to modify. A bare modal or auxiliary opens a predicate with the
# subject elided, so folding it into a series silently gives it the series' subject:
# "source verification will confirm the voltage; and could be useful for verifying
# thickness, but fundamentally lacks novelty" says source verification lacks novelty.
_UNFOLDABLE_HEADS = frozenset(
    {
        "after",
        "although",
        "as",
        "because",
        "before",
        "can",
        "could",
        "given",
        "if",
        "may",
        "might",
        "must",
        "should",
        "since",
        "though",
        "unless",
        "when",
        "whereas",
        "while",
        "will",
        "would",
    }
)


def _folds_into_series(text: str) -> bool:
    """Whether an item can take a conjunction in front of it and still parse."""
    head = re.sub(r"^\W+", "", text).partition(" ")[0].strip(",;:").lower()
    return head not in _UNFOLDABLE_HEADS


def _is_one_sentence(text: str) -> bool:
    """Whether an item can be folded into a series without carrying a break into it."""
    if ";" in text:
        # A conjunct with its own semicolon makes a semicolon series look like it
        # continues past the conjunction, so the reader keeps waiting for the last item.
        return False
    if _PROSE_COLON.search(_ASIDE.sub("", text)):
        # A colon opens material that runs to the end of the sentence, so a conjunct
        # folded in behind one joins what the colon introduced rather than the series.
        # "A direct measurement of the prediction that separates it from the field:
        # coated cells will reach 80% retention 10% later than controls, and evidence
        # for the competing reading its ranking assumes away" read as two things the
        # measurement would show. Such an item is set on its own, where the colon can
        # only reach as far as its own full stop.
        return False
    return not any(
        re.sub(r"^\W+", "", match.group(1).lower()) not in _ABBREVIATIONS
        for match in _SENTENCE_BREAK.finditer(text)
    )


# A parenthesis holds its own punctuation, and a thousands separator is not
# punctuation at all. Neither can be mistaken for the break between two items.
_ASIDE = re.compile(r"\([^()]*\)")
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")

# A colon that punctuates prose, as against one inside a name: PEDOT:PSS is a polymer
# and 1:1 is a ratio, and neither opens anything. Read as punctuation, the first of
# them took "Synthesis of battery-grade, moisture-free PEDOT:PSS" out of its series,
# and the list of required inputs went to the page as "It cannot start until its
# inputs exist: synthesis of ... PEDOT:PSS. Cross-sectional SEM capabilities for
# post-mortem analysis." -- a colon that opens a list of one and a fragment after it.
_PROSE_COLON = re.compile(r":(?=\s|$)")


def _swallows_a_conjunction(text: str) -> bool:
    """Whether a comma in this item could be read as the series separator itself.

    Only a comma in an item that something follows can do that: "generates water,
    potentially causing gas evolution, and anode poisoning" reads as two things
    caused by the water. A comma in the last item has the conjunction already
    behind it and cannot be mistaken for it, and a comma inside "(e.g., via ALD)"
    or inside "1,000" is not a clause break to begin with -- promoting the series
    to semicolons over either put "; and" between two clean phrases.
    """
    return "," in _DIGIT_COMMA.sub("", _ASIDE.sub("", text))


def _series(items: Sequence[str]) -> str:
    """One series, in the separator its own conjuncts allow."""
    if len(items) == 1:
        return items[0]
    # Each item was written as its own sentence, so every one after the first opens
    # with a capital in the middle of a clause unless it is folded in.
    folded = list(items[:1]) + [_spliced(item) for item in items[1:]]
    # Semicolons are for a series whose members already contain commas. Applied to a
    # two-item series of clean clauses -- which is most of them -- "; and" is simply
    # wrong, and it read as though an item had gone missing between them.
    separator = (
        "; " if any(_swallows_a_conjunction(item) for item in folded[:-1]) else ", "
    )
    return separator.join(folded[:-1]) + separator + "and " + folded[-1]


def _names(items: Sequence[str]) -> str:
    """A series of bare noun phrases, which is punctuated unlike a series of clauses.

    ``_series`` is built for stated sentences, where a comma before the conjunction
    separates two independent clauses and belongs there. Applied to a list of two
    words it produced "the correctness, and feasibility reviews" -- a pair
    punctuated as though a third item had been dropped between them.
    """
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _join(items: Sequence[str], *, fallback: str) -> str:
    """Fold a payload list into prose; the reference reports carry no bare stubs."""
    cleaned = [
        " ".join(str(item).split()).rstrip(".")
        for item in items
        if str(item).strip()
        and str(item).strip().lower() not in _STUB_VALUES
        and not _looks_serialised(str(item))
    ]
    if not cleaned:
        return fallback
    # An item that is itself two sentences cannot be a conjunct: folded in, its second
    # sentence starts a new one mid-series, and the "; and ..." that closes the series
    # then hangs off a sentence that never opened a list. Such an item is set on its
    # own, and the items around it keep their series.
    # An item that cannot take a conjunction in front of it is set on its own for the
    # same reason: as its own sentence it keeps the reviewer's wording and reads as
    # the elliptical note it is, where folded in it acquired the series' subject.
    groups: list[list[str]] = [[]]
    for item in cleaned:
        if _is_one_sentence(item) and (not groups[-1] or _folds_into_series(item)):
            groups[-1].append(item)
        else:
            groups.extend(([item], []))
    return " ".join(_sentence(_series(group)) for group in groups if group)


def _review_answer(review: Any) -> str:
    """The verdict in words, including what moved it off its recommendation.

    The number alone is not readable: a three is a revise, but so is an advance held
    at low confidence, and a five-point advance capped at two by a fatal flaw is the
    same integer as a reviewer who simply had too little to go on. Saying which one
    it was is the whole value of printing an answer beside a score.
    """
    verdict = _RECOMMENDATION_WORDS.get(review.recommendation, "revise it first")
    # The figure the reviewer recorded, not a band derived from it. Two things went
    # wrong with bands. The clause used to be inferred from whether the score had moved
    # off its recommendation, and an advance already scores five, so the high-confidence
    # bump had nowhere to go and no advance ever declared its confidence at all. Reading
    # the confidence directly fixed that and exposed the second problem: reviewers in a
    # live run reported 0.75 to 0.95, so "high" was true of forty-six reviews out of
    # forty-seven -- a clause printed that uniformly says nothing, and it flattened a
    # spread the reader can use. The number varies, so the number is what is printed;
    # the debate verdicts on the same page already quote confidence this way.
    stated = (
        f"{verdict}, at the reviewer's stated confidence of {review.confidence:.2f}"
    )
    if review.fatal_flaws and RECOMMENDATION_SCORES.get(review.recommendation, 3) > 2:
        # The cap explains the score; the confidence explains the verdict. Returning
        # early on the cap dropped the confidence from the one review that most needed
        # it -- the only flawed idea on the page said less about itself than the rest.
        return f"{stated}, with the score capped by a fatal flaw"
    return stated


_RECOMMENDATION_WORDS = {
    "advance": "advance the idea as written",
    "revise": "revise it first",
    "insufficient_evidence": "evidence too thin to judge on",
    "reject": "reject it",
}

# The same four verdicts, short enough to be counted in a clause.
_RECOMMENDATION_SHORT = {
    "advance": "advance",
    "revise": "revise",
    "insufficient_evidence": "insufficient evidence",
    "reject": "reject",
}


def _review_score(review: Any) -> int:
    """Fold the recommendation and the reviewer's confidence into the 1-5 scale.

    The fatal-flaw cap is applied last because confidence cannot argue it away. It
    used to be applied first, and the confidence bump then lifted the score straight
    back over it: on a live run every reflection review recorded a fatal flaw at
    confidence 0.90, so fourteen of the eighteen printed as three -- "revise it
    first" -- under a scale the same report explains as capping a flawed review at
    two. A reviewer being sure of a disqualifying finding is a reason to trust the
    cap, not a reason to lift it.
    """
    base = RECOMMENDATION_SCORES.get(review.recommendation, 3)
    if review.confidence >= 0.8 and base < 5:
        base += 1
    elif review.confidence < 0.3 and base > 1:
        base -= 1
    if review.fatal_flaws:
        base = min(base, 2)
    return max(1, min(5, base))


_SECTION_QUESTIONS = {
    "Correctness": (
        "Is the claim correct, and is every load-bearing statement supported by "
        "evidence that has been inspected rather than merely retrieved?"
    ),
    "Novelty": (
        "Does the idea go beyond established practice, and would a specialist in "
        "the field regard the contribution as new?"
    ),
    "Feasibility": (
        "Can the proposed design actually be executed, and would it yield an "
        "interpretable answer under realistic constraints?"
    ),
    "Impact": (
        "Would the result change what practitioners do, and is it worth the cost of "
        "finding out?"
    ),
    "Safety": (
        "Can this work be carried out at all without unacceptable safety, ethical, "
        "or governance exposure, and which approval must precede it?"
    ),
}
# A reviewer-specific question where the reviewer answers a narrower one than its
# section heading implies; otherwise the section question stands.
_REVIEWER_QUESTIONS = {
    "ethics_safety_governance": (
        "Does the idea raise a human-subject, biosafety, privacy, data-rights or "
        "dual-use concern, and which institutional approval must precede it?"
    ),
}


_STRATEGY_POSTURE = {
    "evidence_first": "Evidence-led",
    "mechanism_first": "Mechanism-led",
    "analogy_transfer": "Transfer-led",
    "competing_explanation": "Falsification-led",
}


def _category_path(record: ResearchRecord, candidate: Candidate) -> str:
    """A three-level taxonomy path, widest first, as the reference reports print it."""
    mode = record.session.research_mode.replace("_", " ").title()
    # Clustering runs after evolution, so it names the evolved candidate. Matching on
    # the bare id put five of eight live ideas outside every cluster the same report
    # had just printed them in.
    posture = _STRATEGY_POSTURE.get(
        candidate.generation_strategy, candidate.generation_strategy.title()
    )
    cluster = next(
        (
            cluster.name
            for cluster in (record.landscape.clusters if record.landscape else [])
            for member in cluster.candidate_ids
            if record.ranked_id(member) == record.ranked_id(candidate.id)
        ),
        "",
    )
    # The middle level used to fall back to the generation strategy, which is the
    # same fact the posture beside it is derived from: "Experimental > Mechanism
    # First > Mechanism-led" is one level of taxonomy written twice. An idea no
    # cluster claims has two levels, not a padded three.
    levels = [mode, cluster, posture]
    # A repeated level reads as a rendering fault rather than a taxonomy.
    deduped = [
        level
        for index, level in enumerate(levels)
        if level and level not in levels[:index]
    ]
    return " > ".join(deduped)


# What each field says when the specialist left it empty. They are named rather than
# written inline because the sections below have to be able to tell a stated field from
# an unstated one: "It cannot start until its inputs exist: no external dependency was
# recorded" is a sentence that answers itself, and only the caller knows to avoid it.
_UNSTATED = {
    "Discriminating predictions": "The specialist stated no discriminating prediction.",
    "Alternative explanations": "No competing explanation was recorded for this idea.",
    "Falsifier": "Not stated by the specialist.",
    "Required inputs and dependencies": (
        "No external dependency was recorded for this idea."
    ),
    "Principal risks": "No material risk was recorded for this idea.",
    "Go/no-go tests": "No explicit go/no-go threshold was recorded for this idea.",
}


def _stated(facts: dict[str, str], field_name: str) -> bool:
    """Whether the specialist filled this field in, or the fallback is standing in."""
    return facts.get(field_name) != _UNSTATED[field_name]


def _idea_facts(candidate: Candidate) -> dict[str, str]:
    """The candidate's own fields as prose, the raw material every section draws on."""
    return {
        "Core idea": _sentence(candidate.claim),
        "Mechanism and rationale": _sentence(candidate.rationale),
        "Discriminating predictions": _join(
            candidate.predictions,
            fallback=_UNSTATED["Discriminating predictions"],
        ),
        "Alternative explanations": _join(
            candidate.alternatives,
            fallback=_UNSTATED["Alternative explanations"],
        ),
        "Falsifier": _sentence(candidate.falsifier, fallback=_UNSTATED["Falsifier"]),
        "Required inputs and dependencies": _join(
            candidate.dependencies,
            fallback=_UNSTATED["Required inputs and dependencies"],
        ),
        "Principal risks": _join(
            candidate.risks, fallback=_UNSTATED["Principal risks"]
        ),
        "Go/no-go tests": _join(
            candidate.go_no_go_tests,
            fallback=_UNSTATED["Go/no-go tests"],
        ),
    }


def _revised_form(
    record: ResearchRecord, candidate: Candidate, *, recommended: bool = False
) -> tuple[str, list[tuple[str, str]], list[str]]:
    """The rewrite of this idea as a diff against the form that was ranked.

    Only the fields the rewrite changed are printed. A rewrite typically touches the
    claim and the falsifier and leaves the risks alone, and reprinting all eight would
    put a second copy of the idea under the first with nothing marking which half is
    the news.
    """
    revisions = record.revisions_of(candidate.id)
    if not revisions:
        return "", [], []
    revision = revisions[-1]
    before = _idea_facts(candidate)
    after = _idea_facts(revision.candidate)
    changed = [(label, text) for label, text in after.items() if before[label] != text]
    unchanged = [label for label in after if before[label] == after[label]]
    if not changed:
        return "", [], []
    reviews = record.rereviews_of_latest(candidate.id)
    checked = (
        f" It was re-reviewed after the rewrite: {_review_verdicts(reviews)}."
        if reviews
        else " No re-review of the rewrite is on the record."
    )
    # The diff is against the form that was ranked, so it spans every round of the
    # rewrite, and naming the last round alone told the reader a two-round rewrite
    # happened in one -- with the round-one critiques left off the list it says the
    # rewrite was written to address.
    rounds = sorted({item.round_number for item in revisions})
    addressed = _spliced(
        _join(
            list(
                dict.fromkeys(
                    critique
                    for item in revisions
                    for critique in item.critiques_addressed
                )
            ),
            fallback="the reviews",
        )
    )
    # "to address" hangs off the rewriting, not off the result of it. Appended after
    # the cumulative-result clause it read "evolution rewrote the idea in rounds one,
    # two and three, and this is the cumulative result to address unspecified effect
    # size" -- which says the result was produced in order to address the critique.
    written = (
        f"evolution rewrote the idea in round {_number_word(rounds[0]).lower()} to "
        f"address {addressed}"
        if len(rounds) == 1
        # The critiques are a semicolon series four items long on a live run, and
        # closing it with ", and this is the cumulative result" hung the point of the
        # sentence off the end of a list the reader was still working through. The
        # list ends its own sentence instead.
        else "evolution rewrote the idea in rounds "
        + _names([_number_word(number).lower() for number in rounds])
        + f", and this is the cumulative result. The rounds addressed, between "
        f"them, {addressed}"
    )
    # That only the changed fields are printed is a fact about the layout and the same
    # under every rewritten idea, so it is stated in the preamble above the ideas. What
    # is left here is which fields those are, which differs from rewrite to rewrite.
    lead_in = (
        # Evolution rewrites the whole shortlist; the meta-review recommends a subset
        # of it, or none of it where every candidate carries a fatal flaw. Opening on
        # "the form the meta-review recommends" regardless put a recommendation under
        # four ideas in a report whose section 9 says no idea cleared the bar.
        (
            "This is the form the meta-review recommends"
            if recommended
            else "This is the form evolution produced, which the meta-review does not "
            "recommend"
        )
        + f", and it is not the form ranked above: {written}. "
        + (
            f"The rewrite leaves {_series([label.lower() for label in unchanged])} "
            "unchanged."
            if unchanged
            else "The rewrite changed every field of the idea."
        )
        + checked
    )
    return lead_in, changed, unchanged


def _review_verdicts(reviews: Sequence[CandidateReview]) -> str:
    """The re-review outcomes, counted by verdict."""
    counts = Counter(review.recommendation for review in reviews)
    return _series(
        [
            f"{_plural(count, 'review')} said {_RECOMMENDATION_WORDS.get(name, name)}"
            for name, count in counts.most_common()
        ]
    )


def _attributed_responses(reviews: Sequence[IdeaReview]) -> str:
    """Every recorded response, under the review that wrote it.

    Where every review answered, the count opening the section was "Every review of
    this idea recorded a response." under all eight ideas, above a list that names each
    of those reviews anyway. That the list is complete is a fact about the layout and
    is stated in the preamble; the count survives only where it is short of the
    reviews, which is the case the list cannot show.
    """
    answering = [review for review in reviews if review.rebuttals]
    parts = []
    if len(answering) < len(reviews):
        parts.append(
            f"{_opening(len(answering), 'review')} of this idea recorded a response."
        )
    parts.extend(
        f"The {review.section.lower()} review answered: "
        f"{_join(review.rebuttals, fallback='')}"
        for review in answering
    )
    return " ".join(parts)


def _fatal_flaw_notice(faulted: Sequence[IdeaReview]) -> str:
    """Who recorded a fatal flaw against the idea, and where the finding is printed.

    The subsection is headed Critical Flaws and until this was added it could not
    report one: it keyed off the score, and a score is what a fatal flaw causes rather
    than what it is. So an idea whose correctness review had written the reason it does
    not work printed "the correctness review scored this idea at or below two of five"
    and sent the reader to a list of objections that did not contain the finding.
    """
    flaws = sum(len(review.fatal_flaws) for review in faulted)
    # _joined_titles punctuates a list of idea titles, which are long enough to need
    # semicolons and each carry the word they are a title of. On a list of reviews it
    # wrote "The correctness review; feasibility review; and novelty review", which
    # sets three two-word phrases as though they were clauses and says "review" three
    # times. The Coherence subsection under the same idea already names the same set
    # correctly, so this is the sentence that gave way.
    #
    # Alphabetical order also disagreed with it: the reviews are printed under this
    # idea in the order the run recorded them, and the two sentences listed the same
    # three reviews in two different orders a page apart.
    sections = list(dict.fromkeys(review.section.lower() for review in faulted))
    named = (
        f"{sections[0]} review" if len(sections) == 1 else f"{_names(sections)} reviews"
    )
    return (
        f"The {named} recorded "
        + (
            "a fatal flaw"
            if flaws == 1
            else f"{_number_word(flaws).lower()} fatal flaws"
        )
        # What a fatal flaw is, and that the scale caps a review recording one at two,
        # is stated in the methodology section and again in the preamble above the
        # ideas. Restated here it was the third copy inside one idea's own section,
        # alongside Coherence and the Deep Verification lead-in.
        + " against this idea. "
        + ("It is" if flaws == 1 else "They are")
        + " printed in full under Deep Verification below. Nothing in this run "
        + ("tested it" if flaws == 1 else "tested them")
        + ", and no reviewer withdrew "
        + ("it" if flaws == 1 else "any of them")
        + "."
    )


def _low_score_notice(low: Sequence[IdeaReview]) -> str:
    """Who scored the idea at or below two, and what standing their objection has.

    Reprinting the objections themselves here put a reviewer's assertion into the
    report's own voice, unattributed, and duplicated the Deep verification list a
    page below word for word. Naming the reviews and pointing at that list says the
    same thing once, and says whose judgement it is.
    """
    named = _joined_titles(
        sorted({f"{review.section.lower()} review" for review in low}), fallback=""
    )
    objections = sum(len(review.objections) for review in low)
    opening = (
        f"The {named} scored this idea at or below two of five."
        if named
        else f"{_opening(len(low), 'review')} scored this idea at or below two of five."
    )
    if not objections:
        return (
            f"{opening} Neither recorded an objection to go with the score, so what "
            "the score is a judgement about is not on the record."
            if len(low) > 1
            else f"{opening} It recorded no objection to go with the score, so what "
            "the score is a judgement about is not on the record."
        )
    return (
        f"{opening} What "
        + ("they" if len(low) > 1 else "it")
        # "Deep verification" is not what the heading says. A reader searching the
        # document for the lower-case spelling this pointed them at finds nothing.
        + " objected to is set out under Deep Verification below, attributed to the "
        f"{'reviews' if len(low) > 1 else 'review'} that raised "
        + ("them" if objections > 1 else "it")
        + ". Nothing else in this run has tested "
        + ("those objections" if objections > 1 else "that objection")
        + ", so "
        + ("they stand" if objections > 1 else "it stands")
        + " unresolved rather than established."
    )


def _conclusion(
    facts: dict[str, str],
    reviews: Sequence[IdeaReview],
    *,
    shortlisted: bool,
    accepted_flaw: AdjudicationNote | None,
) -> str:
    """What to do about this idea next, and what doing it will not settle.

    The falsifier is quoted under Description and the thresholds under Feasibility
    Assessment, so neither is reprinted here; what this slot is for is the order the
    two go in, which is nowhere else in the report. But order of work is the same
    sentence for every idea that recorded both, and it was the whole of the section:
    eight conclusions in two variants, none of them about the idea above them.

    What is on the record and is about this idea is where it is weakest. The reviews
    are printed below in section order, and the reader has to compare five scores to
    find the low one; the lead-in to them reports the span without saying which
    dimension sits at the bottom of it.
    """
    # An idea the shortlist dropped has no next move, and asserting one and withdrawing
    # it two clauses later -- "The next move is the go/no-go work ... It is not on the
    # shortlist, so nothing is scheduled against it" -- was the conclusion of four of
    # eight ideas. The condition comes first, and what follows it is conditional.
    withheld = (
        ""
        if shortlisted
        else "It is not on the shortlist, so nothing is scheduled against it. "
    )
    lead = (
        "The next move is " if shortlisted else "If it is revived, the first move is "
    )
    # Why the go/no-go comes before the falsifier is true of every idea that records
    # both, and it opened all eight conclusions in the same twenty words. It is stated
    # once in the preamble above the ideas; what is left here is which of the two this
    # idea actually recorded.
    order_of_work = (
        f"{lead}the go/no-go work under Feasibility Assessment above, and the "
        "falsifier under Description after it."
        if _stated(facts, "Go/no-go tests") and _stated(facts, "Falsifier")
        else f"{lead}the go/no-go work under Feasibility Assessment above. No "
        "falsifier was recorded for this idea, so that check is the only stated "
        "result that would stop the work."
        if _stated(facts, "Go/no-go tests")
        else f"{lead}the check under Description, which is the only one the "
        "specialist set down against its own idea; no go/no-go threshold was "
        "recorded to come before it."
        if _stated(facts, "Falsifier")
        else "The record states neither a go/no-go threshold nor a "
        "falsifier for this idea, so there is no next move in it: what to "
        "run first would have to be decided outside this report."
    )
    scores = [review.score for review in reviews]
    if not scores:
        weakest = ""
    elif min(scores) == max(scores):
        weakest = (
            f" All {_plural(len(scores), 'review')} of this idea scored it "
            f"{scores[0]} of five, so no one dimension of it is weaker than "
            "the rest to attend to first."
        )
    else:
        floor = min(scores)
        # _names, not _joined_titles: that helper punctuates three or more with
        # semicolons, which is right for multi-word idea titles and reads as a list
        # of headings when applied to "correctness; feasibility; and novelty".
        #
        # In the order the run recorded the reviews, which is the order they are
        # printed in below and the order the Coherence subsection names them in. Sorted
        # alphabetically, this sentence and that one listed the same three reviews in
        # two different orders on the same page.
        sections = list(
            dict.fromkeys(
                review.section.lower() for review in reviews if review.score == floor
            )
        )
        weakest = (
            f" The lowest score it received is {floor} of five, from the "
            f"{sections[0]} review, printed in full below: that is the review "
            "to read before any of the above is commissioned."
            if len(sections) == 1
            # "all printed in full below" over a pair: "all" counts three or more,
            # and the sentence names exactly the two reviews it is talking about.
            else f" Its lowest score, {floor} of five, is shared by the "
            f"{_names(sections)} reviews, "
            + ("both" if len(sections) == 2 else "all")
            + " printed in full below: they are what to read before any of the "
            "above is commissioned."
        )
    # The bench work is a test of the idea. An accepted fatal flaw is not something a
    # test can return a verdict on -- it was allowed to stand by a person, and it goes
    # on standing whatever the cells do. Saying what the next move reaches has to say
    # what it does not reach, or the section reads as a plan that disposes of the
    # objection above it.
    accepted = (
        " None of that work reaches the fatal flaw accepted above: the flaw was "
        "allowed to stand rather than answered, so no result from these checks "
        "removes it."
        if accepted_flaw
        else ""
    )
    return f"{withheld}{order_of_work}{weakest}{accepted}"


def _motivation(facts: dict[str, str], supporting: Sequence[str]) -> str:
    """What the report holds in favour of this idea, and nothing that holds of all.

    The mechanism is under Description a few hundred words above, so it is not
    reprinted here; what this slot is for is the evidence the idea's own proposal
    cites, which is on the record one idea at a time and appeared nowhere else in the
    report. Around it stood two variants of one sentence -- that a discriminating
    prediction comes out one way if the mechanism holds and another if the competing
    reading does -- which is true of every idea that states one and was the whole of
    this section for seven of seven ideas on a live run. That argument is in the
    preamble above the ideas. What is left here is which of the two this idea has.
    """
    cited = (
        f"The findings this idea cites: {_join(list(supporting), fallback='none.')} "
        if supporting
        else "No finding in this report's evidence is cited for this idea. "
    )
    if not _stated(facts, "Discriminating predictions"):
        return cited + (
            "No discriminating prediction was stated for it"
            + ("" if supporting else " either")
            + ", so there is nothing here that a result could confirm or refute, and "
            "the case for the idea rests on its mechanism alone."
        )
    # That no result in this run tested the prediction is true of every prediction in
    # the report -- the run proposes work rather than doing any -- so it is stated in
    # the preamble above the ideas rather than under each of the seven ideas whose
    # case rests on one.
    return cited + (
        ("The rest of the case " if supporting else "The case ")
        + "for it is the discriminating prediction under Description."
    )


def _novelty_standing(review: IdeaReview | None, field: Sequence[int]) -> str:
    """The novelty score, placed against the field the same reviewers scored.

    This section was the score and nothing else -- "The novelty review scored this 5 of
    five." under all eight ideas -- and the review table a few lines below prints that
    figure again. Where the score sits in the run is not in the table and is what the
    score is for: five of five is a different statement in a field that tops out at
    three from one where half the ideas also scored five.
    """
    if review is None:
        return "No novelty review was recorded for this idea."
    stated = f"The novelty review scored this {review.score} of five"
    if len(field) < 2 or min(field) == max(field):
        return stated + (
            ", as it scored every other idea in this run, so the score separates this "
            "one from none of them."
            if len(field) > 1
            else "."
        )
    top, bottom = max(field), min(field)
    if review.score == top:
        place = "the highest any idea in this run received"
    elif review.score == bottom:
        place = "the lowest any idea in this run received"
    else:
        return f"{stated}, inside a field running from {bottom} to {top}."
    shared = field.count(review.score) - 1
    return f"{stated}, {place}" + (
        f", shared with {_plural(shared, 'other idea')}."
        if shared
        else ", and no other matched it."
    )


def _summary_sections(
    facts: dict[str, str],
    reviews: Sequence[IdeaReview],
    *,
    rank: int,
    elo: int,
    shortlisted: bool,
    accepted_flaw: AdjudicationNote | None = None,
    tied_with: int = 0,
    tie_straddles_cut: bool = False,
    supporting: Sequence[str] = (),
    novelty_field: Sequence[int] = (),
) -> dict[str, str]:
    """The eight fixed Summary subsections the reference reports print per idea.

    They are a verdict on the idea rather than a restatement of it, so each one folds
    the candidate's own fields together with what the reviewers concluded.
    """
    scores = [review.score for review in reviews]
    mean = sum(scores) / len(scores) if scores else 0.0
    low = [review for review in reviews if review.score <= 2]
    faulted = [review for review in reviews if review.fatal_flaws]
    # A review that recorded a fatal flaw is already reported by the sentence above
    # the score one: repeating "and it also scored the idea at or below two" of the
    # same review says the cap twice and calls it two findings.
    scored_low_only = [review for review in low if not review.fatal_flaws]
    novelty = next((review for review in reviews if review.section == "Novelty"), None)
    feasibility = next(
        (review for review in reviews if review.section == "Feasibility"), None
    )
    verdict = (
        "carried forward onto the shortlist"
        if shortlisted
        else "held back from the shortlist"
    )
    return {
        # The claim is printed in full under Idea Proposal, immediately above this
        # block. Opening the verdict with it again restated a paragraph the reader had
        # just finished, and buried the verdict the section exists to deliver.
        "Executive Verdict": (
            f"This idea finished rank {rank} on an Elo of {elo}"
            # A tie the shortlist cut runs through is the one case where the verdict
            # is not a decision at all. Two live ideas finished on the same rating
            # with the same mean review score and were printed as carried forward and
            # held back, each stated flatly, with the tie noted only in a rank line
            # some thirty lines above. What separated them was the tie-break the sort
            # happened to use, and the sentence a reader acts on has to say so.
            + (
                " — tied with "
                + (
                    "another idea"
                    if tied_with == 1
                    # "tied with 2 others" put a digit in prose beside "averaging 4.2
                    # of five across five reviews" in the same sentence, where the
                    # measured mean is a figure and the counts are words.
                    else _plural(tied_with, "other")
                )
                + ", which the shortlist cut runs through — and was "
                if tie_straddles_cut
                else " and was "
            )
            + f"{verdict}"
            + (
                # "3.6 of five across 5 reviews" spells one small number and prints
                # the other as a digit inside a single clause. The mean is data and
                # stays a figure; the two counts around it are prose.
                f", averaging {mean:.1f} of five across "
                f"{_number_word(len(scores)).lower()} "
                + ("review." if len(scores) == 1 else "reviews.")
                if scores
                else ", with no review recorded against it."
            )
        ),
        # An accepted fatal flaw leads this subsection whatever else was raised: it is
        # the one objection the reader knows was neither answered nor removed.
        #
        # What followed it used to be the low-scoring reviews' objections reprinted
        # bare, in the report's own voice and with no attribution. One of them read
        # "The core rationale is falsified by the very evidence it cites" -- an
        # assertion that a source nobody in the run had opened settles the idea,
        # printed as a finding four lines under a notice saying none of this idea's
        # evidence had been verified. The same sentences are then printed again,
        # correctly attributed, under Deep verification. So this says who scored the
        # idea down, what standing their objection has, and where to read it.
        #
        # Three claims in descending order of standing, each printed only if it holds:
        # a flaw a human accepted, a flaw a reviewer recorded, a score with no flaw
        # behind it. The middle one was missing entirely, which is how a subsection
        # headed Critical Flaws came to report scores and nothing else.
        "Critical Flaws": " ".join(
            item
            for item in (
                # The chapter head printed this idea's accepted flaw in full a page
                # above, so here it is recalled rather than quoted a second time.
                accepted_flaw.reprise if accepted_flaw else "",
                _fatal_flaw_notice(faulted) if faulted else "",
                _low_score_notice(scored_low_only) if scored_low_only else "",
            )
            if item
        )
        # What a clean subsection is and is not evidence of holds wherever it is
        # printed, and it was printed under four of eight ideas. It is in the preamble
        # above the ideas; here the fact is enough.
        or "No reviewer recorded a fatal flaw against this idea.",
        # The heading is the reference layout's. Nothing validated these: they are the
        # risks the proposing specialist listed about its own idea, and printing them
        # bare under that heading told the reader they had been through something.
        # Whose risks these are is stated in the preamble above the deep dives, which
        # is where it can be said once instead of at the head of all eight.
        "Identified issues & Validated Risks": (
            facts["Principal risks"]
            if _stated(facts, "Principal risks")
            else "The specialist that proposed this idea named no risk against it, "
            "and no reviewer was asked to supply one. An empty list here is a gap in "
            "the record rather than a finding about the idea."
        ),
        # The heading is the reference layout's and is kept, but what sits under it
        # is a review's list of responses, not a list of objections it disposed of.
        # Several of them concede: "the reviewer is correct that no power calculation
        # was given" was printed under Addressed Objections as though the objection
        # had been dealt with. The record does not say which response answers which
        # objection either, so the caption has to stop short of both claims.
        #
        # Pooling five reviews' responses into one series also pooled their subjects.
        # A novelty response written as "Could be useful for verifying the exact
        # thickness dependence, but fundamentally lacks novelty" has its subject
        # elided, so folded in after the correctness response it read as a claim that
        # source verification lacks novelty. Each response is attributed to the review
        # that wrote it, which is both the true thing to say and the only arrangement
        # in which an elided subject can be recovered by the reader.
        "Addressed Objections": (
            _attributed_responses(reviews)
            if any(review.rebuttals for review in reviews)
            else "No review of this idea recorded a response of any kind, so every "
            "objection raised against it still stands as written."
        ),
        # The mechanism is stated in full under Description, a few hundred words
        # above. Reprinting it here and again under Deep Verification put the same
        # paragraph into one idea's section three times, which is what a summary is
        # supposed to prevent. This section names what the mechanism buys instead.
        #
        # It used to open by saying where the mechanism was and close by saying what
        # a specific prediction is worth. Both are true of every idea in the report,
        # so both are in the preamble above the deep dives, and what is left is the
        # one clause that is about this idea.
        #
        # The predictions themselves were then spliced in here in full, which is the
        # duplication this section was written to remove: Description prints them a
        # page above, and the two copies differed only in the case of the first
        # letter. And "what earns it a place in the ranking" was printed under every
        # idea including the ones the ranking put last and the shortlist dropped,
        # asserting a merit the section below it withdraws.
        #
        # The two variants of the sentence below were the whole of this section for
        # all eight ideas: a heading promising evidence, under which no finding this
        # idea cites was ever named. The findings are on the record, one idea at a
        # time, and they are what the heading is for.
        "Supporting Arguments & Evidence (Motivation)": _motivation(facts, supporting),
        # What a novelty score is a judgement about is the same for all eight ideas
        # and is stated in the preamble, so this is the score and where it sits.
        "Goal Alignment & Novelty": _novelty_standing(novelty, novelty_field),
        "Feasibility Assessment (Go/No-Go Decision)": (
            (
                f"The feasibility review scored this {feasibility.score} of five. "
                if feasibility
                else "No feasibility review was recorded for this idea. "
            )
            + (
                "It cannot start until its inputs exist: "
                f"{_spliced(facts['Required inputs and dependencies'])}. "
                if _stated(facts, "Required inputs and dependencies")
                else "No input or dependency was recorded for it, so nothing here "
                "states what has to be in place before the work can start. "
            )
            + (
                # This used to promise "a stated threshold" and then print however
                # many the specialist wrote, which was routinely three or four. The
                # count is not what the reader needs, so the sentence stops making
                # a claim about it. What the tests are for is then the same clause
                # under every idea that recorded any, so it is in the preamble above
                # the ideas and this prints the tests alone.
                f"Its go/no-go tests: {_spliced(facts['Go/no-go tests'])}."
                if _stated(facts, "Go/no-go tests")
                else "No go/no-go threshold was recorded either, so nothing states "
                "in advance what result would stop the work."
            )
        ),
        # The falsifier is quoted under Description. Quoting it again here, a page
        # below, printed several sentences twice inside one idea's section. What
        # replaced it was a global caveat -- until the check is run this is a
        # proposal, and nothing here is a finding -- which is true of every idea in
        # the report and was therefore printed identically under all eight of them.
        # What is left is in _conclusion, which the docstring there explains.
        "Conclusion": _conclusion(
            facts,
            reviews,
            shortlisted=shortlisted,
            accepted_flaw=accepted_flaw,
        ),
    }


def _table_rows(candidate: Candidate) -> list[tuple[str, str]]:
    """The per-idea comparison grid, using one row-label set across the whole report."""
    values = {
        "Mechanism": _sentence(candidate.rationale),
        "Discriminating prediction": _join(
            candidate.predictions[:1], fallback="None recorded."
        ),
        "Falsifier": _sentence(candidate.falsifier),
        "Key dependency": _join(candidate.dependencies[:1], fallback="None recorded."),
        "Principal risk": _join(candidate.risks[:1], fallback="None recorded."),
    }
    return [(label, values[label]) for label in IDEA_TABLE_ROWS]


# Every specialist as a reader would name it, rather than as the run files it. These
# are the words that go into a warning about a stage; the reviewer lead-ins below name
# the person rather than the pass, which reads differently over a review.
_AGENT_NAMES = {
    "goal_manager": "goal scoping",
    "evidence_discovery": "literature discovery",
    "source_verification": "source verification",
    "generation": "idea generation",
    "reflection": "evidence and correctness review",
    "novelty_review": "novelty review",
    "methods_statistics": "methods and statistics review",
    "ethics_safety_governance": "ethics, safety and governance review",
    "impact_review": "impact review",
    "ranking": "tournament ranking",
    "evolution": "evolution of the shortlist",
    "proximity": "clustering by mechanism",
    "meta_reviewer": "meta-review",
}

# The reviewers as a reader would name them, rather than as the pipeline files them.
_REVIEWER_NAMES = {
    "reflection": "Evidence and correctness reviewer",
    "novelty_review": "Novelty reviewer",
    "methods_statistics": "Methods and statistics reviewer",
    "impact_review": "Impact reviewer",
    "ethics_safety_governance": "Ethics, safety and governance reviewer",
}


def _reviewer_lead_in(reviewer: str) -> str:
    """Name the reviewer in words rather than in the id the pipeline files it under.

    "Ethics safety governance review:" is an enum with its underscores taken out and
    it reads as one -- three nouns in a row, no conjunction, and a heading the reader
    can see was not written by anyone. "Reflection review:" is worse, because
    reflection is the stage's internal name for reviewing and says nothing about what
    that reviewer looked at. Known reviewers are named; anything else at least gets
    its conjunction and the person doing the work rather than the pass.
    """
    known = _REVIEWER_NAMES.get(reviewer)
    if known:
        return f"{known}:"
    words = [
        word
        for word in reviewer.replace("_", " ").split()
        if word.lower() not in {"review", "reviewer"}
    ]
    if not words:
        return "Specialist reviewer:"
    name = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    return f"{name.capitalize()} reviewer:"


def _idea_reviews(record: ResearchRecord, candidate_id: str) -> list[IdeaReview]:
    """Every review of one idea, bucketed into the four scored section names.

    Deduplication is by what the review is, not by its id. Each specialist
    numbers its own set from ``rev_001``, so an id-keyed guard treated the
    novelty, feasibility, impact and safety reviews of a candidate as repeats
    of its correctness review and dropped them: five reviews per idea reached
    the report as one or two, and every idea in a live run was printed with
    "No feasibility review was recorded" over a feasibility review that existed.
    """
    reviews: list[IdeaReview] = []
    seen: set[tuple[str, str]] = set()
    for review_set in record.reviews:
        for review in review_set.reviews:
            key = (review.criterion, review.reviewer)
            if review.candidate_id != candidate_id or key in seen:
                continue
            seen.add(key)
            score = _review_score(review)
            section = CRITERION_SECTIONS.get(review.criterion, "Correctness")
            reviews.append(
                IdeaReview(
                    section=section,
                    lead_in=_reviewer_lead_in(review.reviewer),
                    question=_REVIEWER_QUESTIONS.get(
                        review.reviewer, _SECTION_QUESTIONS[section]
                    ),
                    findings=[_sentence(item) for item in review.findings],
                    objections=[_sentence(item) for item in review.objections],
                    rebuttals=[_sentence(item) for item in review.rebuttals],
                    fatal_flaws=[_sentence(item) for item in review.fatal_flaws],
                    # The reference reports print a matched pair, the verdict beside
                    # the number. Setting both to the score printed "Answer: 3" over
                    # "Score: 3" under every scored review in the report, which tells
                    # the reader the same thing twice and never says what 3 means.
                    answer=_review_answer(review),
                    score=score,
                    recommendation=review.recommendation,
                )
            )
    order = {name: index for index, name in enumerate(REVIEW_SECTIONS)}
    return sorted(reviews, key=lambda review: order.get(review.section, len(order)))


# What a spread means, and what settles one, are the same statements under every idea
# that has one -- three sentences of standing explanation printed seven times on a live
# run, wrapped around one clause of idea-specific fact. The explanations are collected
# under the ideas that raise them and printed once above them all; what stays in place
# is this idea's spread, this idea's lowest review, and this idea's falsifier.
COHERENCE_SPREAD_NOTE = (
    "Where the reviews of an idea disagree by more than a point, the idea is strong "
    "on one dimension and weak on another rather than uniformly graded, and the "
    "review sections under it are to be read separately rather than averaged."
)
COHERENCE_EVIDENCE_NOTE = (
    "Where the lowest review of an idea is the evidence and correctness review, the "
    "disagreement is about the grounding rather than the design, and reading the "
    "cited sources in full is what would settle it. The falsifier would still decide "
    "the idea, but not until there is agreement on what the idea is built on."
)
COHERENCE_FALSIFIER_NOTE = (
    "Where a falsifier is recorded, that is what settles a disagreement: the "
    "specialist set it down against its own idea, and it names a result that would "
    "end the idea rather than amend it, which is what makes the reviews above it "
    "decidable at all."
)
# Printed in this order whichever idea first raised them, so that the general case
# precedes the exception it is qualified by.
_COHERENCE_NOTES = (
    COHERENCE_SPREAD_NOTE,
    COHERENCE_EVIDENCE_NOTE,
    COHERENCE_FALSIFIER_NOTE,
)


def _coherence(
    reviews: Sequence[IdeaReview], facts: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Whether the reviews agree with each other, which no single score reveals.

    Returns the paragraphs for this idea and the standing explanations they rely on,
    which the caller prints once above every idea rather than under each.
    """
    if not reviews:
        return [
            "No review was recorded against this idea, so there is no agreement to "
            "assess and its position rests on the tournament alone."
        ], []
    scores = [review.score for review in reviews]
    spread = max(scores) - min(scores)
    # A narrow spread is agreement about how good the idea is, which is not agreement
    # about what to do with it: five reviews inside one point closed as four revisions
    # and one advance, and the report called that unanimous. The recommendations are
    # counted separately because they are the half a reader acts on.
    asked = Counter(
        review.recommendation for review in reviews if review.recommendation
    )
    verdicts = ", ".join(
        f"{count} {_RECOMMENDATION_SHORT.get(name, name)}"
        for name, count in asked.most_common()
    )
    notes: list[str] = []
    if spread > 1:
        # A clause rather than a sentence: "span 2 to 5 of five" and "they disagree by
        # more than a point" are one fact, and set as two sentences the second read as
        # a second finding. It is kept in the words the explanation above the ideas
        # uses, so the reader can see which rule this idea has just met.
        agreement = "a disagreement of more than a point."
        notes.append(COHERENCE_SPREAD_NOTE)
    elif len(asked) > 1:
        agreement = (
            "The scores agree, but the recommendations do not "
            f"({verdicts}), so the spread understates the disagreement about what to "
            "do next."
        )
    else:
        agreement = (
            "They agree, so the idea presents the same way from every angle that was "
            "examined."
        )
    # A fatal flaw caps its review at two, so a field of capped reviews agrees, and the
    # line reporting that agreement read as a consensus about the idea's quality: "they
    # agree, so the idea presents the same way from every angle that was examined",
    # printed over a review that had called the idea disqualified. Agreement among
    # reviews one of which is a disqualification is not the same kind of fact.
    flawed = [review for review in reviews if review.fatal_flaws]
    if flawed:
        recorded = sum(len(review.fatal_flaws) for review in flawed)
        sections = list(dict.fromkeys(review.section.lower() for review in flawed))
        records = (
            (
                f"the {sections[0]} review records "
                if len(sections) == 1
                else f"the {_names(sections)} reviews record "
            )
            + ("a fatal flaw" if recorded == 1 else _plural(recorded, "fatal flaw"))
            + " against the idea"
        )
        # Written for the agreement branches and appended to all three, this printed
        # "They disagree by more than a point ... That agreement is not a clearance"
        # over three ideas. What a recorded flaw does to a spread is not what it does
        # to a consensus, so the two cases say different things. That the scale caps
        # such a review at two is in the methodology section and again in the preamble
        # above the ideas, so it is not said a third time here.
        agreement += (
            f" Part of that spread is a disqualification rather than a grade: {records}."
            if spread > 1
            else f" That agreement is not a clearance: {records}, and nothing in this "
            "run answered it."
        )
    lines = [
        # The count is prose and the scores are data, which is the convention the
        # Executive Verdict twelve lines above already follows: it wrote "across
        # five reviews" while this wrote "The 5 reviews", in one section.
        f"The {_number_word(len(reviews)).lower()} "
        + ("review" if len(reviews) == 1 else "reviews")
        + f" of this idea span {min(scores)} to {max(scores)} of five"
        + (", " if spread > 1 else ". ")
        + agreement
    ]
    # Counted per review, not per objection. Subtracting the two list lengths asserted
    # a pairing the record does not carry, and the difference it produced was reported
    # as a number of unanswered objections that no reviewer had left unanswered.
    silent = [
        review
        for review in reviews
        if review.objections and not any(item.strip() for item in review.rebuttals)
    ]
    if silent:
        objections = sum(len(review.objections) for review in silent)
        lines.append(
            f"{_opening(len(silent), 'review')} raised "
            f"{_plural(objections, 'objection')} and recorded no response to "
            + ("it" if objections == 1 else "any of them")
            + f" — the {_joined_titles([review.section for review in silent])} "
            + ("review" if len(silent) == 1 else "reviews")
            + ". An objection nobody answered is not a refuted one, and "
            + ("it still applies" if objections == 1 else "each still applies")
            + " to the idea as written."
        )
    # This used to quote the falsifier, which Description states and the Conclusion
    # restated, putting the same several sentences into one idea's section three
    # times. It also opened by asserting that the claim and the falsifier were
    # consistent with one another, which nothing in the run had checked.
    #
    # What replaced it then claimed, under every idea alike, that the falsifier is
    # what would settle the disagreement -- an idea-specific claim made with no
    # idea-specific input. Where the split is driven by the evidence review, it is
    # false: no falsifier settles a dispute about whether a cited source says what
    # the idea says it says. Which review sits at the bottom is on the record, so
    # the sentence is decided from that rather than asserted over it.
    lowest = min(reviews, key=lambda review: review.score)
    # Whether there is a disagreement for a falsifier to settle. The sentence below
    # asserted one unconditionally, so on an idea whose reviews had just been
    # reported as agreeing it named "the disagreement between the reviews above"
    # two lines under "They agree". Where they agree, what a falsifier is for is
    # testing the reading they share.
    disputed = spread > 1 or len(asked) > 1
    if spread > 1 and lowest.section == "Correctness":
        lines.append(
            "The lowest of them is the evidence and correctness review, at "
            f"{lowest.score} of five: what it faults is the grounding, not the "
            "experiment."
        )
        notes.append(COHERENCE_EVIDENCE_NOTE)
    elif _stated(facts, "Falsifier"):
        lines.append(
            "The falsifier stated under Description is what would settle the "
            "disagreement between the reviews above."
            if disputed
            else "The falsifier stated under Description is what would put the "
            "reading they share to the test."
        )
        notes.append(COHERENCE_FALSIFIER_NOTE)
    else:
        lines.append(
            "No falsifier was recorded for this idea, so there is no stated result "
            + (
                "that would settle the disagreement between the reviews above. Each "
                "of them is a reading of a claim nothing can yet be held against."
                if disputed
                else "that would test the reading the reviews share. Their agreement "
                "is about a claim nothing can yet be held against."
            )
        )
    return lines, notes


def _objections_raised(reviews: Sequence[IdeaReview]) -> list[tuple[str, str, bool]]:
    """Each objection, the section that raised it, and whether that review responded.

    A reviewer writes objections and rebuttals as two independent lists. Nothing in
    the contract ties the n-th rebuttal to the n-th objection, and on a live run they
    did not line up: a rebuttal about statistical power was printed as the answer to
    an objection about blinding, while the power objection three items later was
    declared unanswered. Both statements were manufactured by the zip. What the record
    does support is which review raised an objection and whether that review recorded
    any response at all, so that is all this reports.
    """
    raised: dict[str, tuple[str, bool]] = {}
    for review in reviews:
        responded = any(item.strip() for item in review.rebuttals)
        for objection in review.objections:
            # The same objection raised twice keeps whichever mention drew a response.
            if objection not in raised or (responded and not raised[objection][1]):
                raised[objection] = (review.section, responded)
    return [
        (section, objection, responded)
        for objection, (section, responded) in raised.items()
    ]


def _fatal_flaws_raised(reviews: Sequence[IdeaReview]) -> list[tuple[str, str]]:
    """Each fatal flaw and the section that recorded it, deduplicated by text.

    Two reviews reaching the same fatal flaw is one finding about the idea, not two,
    and printing it twice under two headings reads as independent corroboration. The
    first section to record it keeps it; that both reached it is visible from the
    scores, which the cap puts at two of five for each of them.
    """
    raised: dict[str, str] = {}
    for review in reviews:
        for flaw in review.fatal_flaws:
            raised.setdefault(flaw, review.section)
    return [(section, flaw) for flaw, section in raised.items()]


# What the section says when no reviewer objected to the idea at all. The lead-in
# defined above the idea brief promises objections, so it cannot introduce a list that
# has none.
DEEP_VERIFICATION_STANDING_LEAD_IN = (
    "No review of this idea recorded an objection against it, so what follows is the "
    "standing check rather than anything a specialist raised. An idea nobody objected "
    "to has not thereby been verified."
)


def _standing_checks(facts: dict[str, str]) -> list[tuple[str, str]]:
    """The checks that apply to any idea, printed only where nothing else was raised.

    These three used to be appended to every idea's list. Byte-identical across eight
    ideas, they printed twenty-four paragraphs that said the same thing, immediately
    under the objections that were the reason to read the section -- and a reader who
    has met them twice stops reading the third. They earn their place only where the
    list would otherwise be empty.
    """
    checks = [
        (
            "Independent Confirmation of the Mechanism",
            "The mechanism set out under Description has to survive a test that does "
            "not presuppose the claim it was invoked to support. Every result below "
            "it inherits whatever that test finds.",
        )
    ]
    if _stated(facts, "Discriminating predictions") and _stated(
        facts, "Alternative explanations"
    ):
        checks.append(
            (
                "Discriminating Power of the Prediction",
                "The prediction has to be one the competing explanation set out under "
                "Description does not also make. A shared prediction cannot settle "
                "anything between them, however cleanly it comes out.",
            )
        )
    if _stated(facts, "Required inputs and dependencies"):
        checks.append(
            (
                "Existence of the Stated Dependencies",
                "Each input this idea depends on has to be confirmed to exist before "
                "effort is committed, not after. They are listed under Feasibility "
                "Assessment above, and none of them has been checked here.",
            )
        )
    return checks


def _deep_verification(
    reviews: Sequence[IdeaReview], facts: dict[str, str]
) -> tuple[str, list[tuple[str, str]]]:
    """The numbered flaw list: what would have to be checked before believing this."""
    # The heading used to be derive_idea_title(objection), which is the first seven
    # words of the sentence the body then prints in full: "No Evidence Provided to
    # Support the HF" over "No evidence provided to support the HF scavenging
    # mechanism ...". It restated the body and truncated it mid-phrase doing so. What
    # the reader cannot get from the body at a glance is which review raised the
    # objection and whether anyone answered it, so the heading says that instead.
    #
    # The body is the objection and nothing else. Each item used to carry its own copy
    # of the pairing caveat -- "that review did record responses; the record does not
    # say which of them was meant for this objection" -- so one sentence of objection
    # arrived under three sentences of boilerplate identical to the item above it, and
    # forty-one objections printed the same two paragraphs alternating down the page.
    # The caveat belongs in the lead-in, where it is stated once.
    #
    # The fatal flaws lead the list. They used to appear nowhere in the report at all:
    # a reviewer would write "the cited source measures a different chemistry, so the
    # rationale does not hold", the scale would cap that review at two, and the reader
    # was told only that the review had scored the idea low and sent here -- to a list
    # that printed the objections and dropped the finding the score was actually about.
    fatal = [
        (
            f"Fatal flaw recorded by the {section.lower()} review",
            _not_opened_on_a_numeral(flaw, kind="flaw"),
        )
        for section, flaw in _fatal_flaws_raised(reviews)
    ]
    checks = fatal + [
        (
            f"Raised by the {section.lower()} review"
            + ("" if responded else ", unanswered"),
            _not_opened_on_a_numeral(objection, kind="objection"),
        )
        for section, objection, responded in _objections_raised(reviews)
    ]
    if fatal:
        return DEEP_VERIFICATION_FATAL_LEAD_IN, checks
    if checks:
        return DEEP_VERIFICATION_LEAD_IN, checks
    return DEEP_VERIFICATION_STANDING_LEAD_IN, _standing_checks(facts)


def _not_opened_on_a_numeral(text: str, *, kind: str) -> str:
    """A recorded objection set as a paragraph, without a digit for its first character.

    Prose does not open a sentence on a numeral -- "15 nm might still be thin enough to
    allow tunneling or sufficient lithiation" reads as a list item that lost its bullet.
    The number cannot be spelled here the way a count the report itself wrote is: this
    is a measurement, and it is the specialist's sentence, so putting a noun in front of
    it ("A 15 nm layer might still be...") would be the report deciding what the
    reviewer meant. A lead-in carries the sentence and leaves every word of it alone.
    """
    stated = _sentence(text)
    return f"The {kind} is that {stated}" if stated[0].isdigit() else stated


_HYPOTHESIS_LABEL = re.compile(r"\b(?:Hypothesis|H)\s?([12])\b")

# The same two slots named by position rather than by number. A judge wrote "avoids
# the fundamental material instability that plagues the first proposal", which the
# label pattern does not match, so the sentence was reprinted unchanged on both
# ideas' pages -- one of them beside "this idea" and the other beside "the opposing
# idea", with nothing on either page saying which proposal came first. "The former"
# and "the latter" fail the same way and are worse: they refer to an ordering the
# reprint has already dissolved.
_POSITIONAL_LABEL = re.compile(
    r"\bthe (?:(first|second) (?:proposal|hypothesis|idea|approach)|(former|latter))\b",
    re.IGNORECASE,
)
_POSITIONAL_SLOTS = {"first": "1", "former": "1", "second": "2", "latter": "2"}

# A debater that introduces a position often glosses it once -- "Hypothesis 1 (H1)"
# -- and both halves are the same label, so both were substituted and the reprint
# read "This idea (this idea)". The gloss carries nothing the name does not.
_GLOSSED_LABEL = re.compile(r"\s*\((?:the )?(?:this|opposing) idea\)")


# What decided a match, as a noun phrase. The stored value is a snake_case enum;
# printing it straight put "llm comparison" into eighteen table rows of one report,
# and de-underscoring it was not enough -- a run judged without a model printed
# "deterministic" as the name of a thing, and the sentence around it read "settled
# in a single pass by deterministic rather than by argued rounds".
_JUDGE_LABELS = {
    "deterministic": "an arithmetic score comparison",
    "llm_comparison": "a single-pass model comparison",
    "llm_debate": "a multi-turn model debate",
}


# The same three, as a table cell rather than as a clause. "Judge: a single-pass
# model comparison" reads as a sentence that lost its verb inside a narrow column.
_JUDGE_COLUMN = {
    "deterministic": "Arithmetic",
    "llm_comparison": "Model, single pass",
    "llm_debate": "Model, debated",
}


def _judge_label(judge: str) -> str:
    """How the match was decided, as a noun phrase a sentence can take."""
    return _JUDGE_LABELS.get(
        judge,
        " ".join(word.upper() if word == "llm" else word for word in judge.split("_")),
    )


def _judge_column(judge: str) -> str:
    """The same, short enough for the Judge column of a match table."""
    return _JUDGE_COLUMN.get(judge, _judge_label(judge))


def _sided(turn: str, *, first: bool) -> str:
    """A debate turn read from one side of the match rather than from above it.

    The debaters write "Hypothesis 1" and "Hypothesis 2", which are the two slots of
    the presentation, not names. Which idea occupies slot one alternates by match --
    that alternation is the tournament's position-bias control -- so the slot cannot
    be read off ``candidate_a_id``, only off ``presented_first_id``. Reading it off
    the former printed half of every debate from the wrong side: the transcript's own
    conclusion then contradicted the verdict line under it, which is worse than
    leaving the slots unresolved.
    """
    mine = "1" if first else "2"
    sided = _HYPOTHESIS_LABEL.sub(
        lambda match: "this idea" if match.group(1) == mine else "the opposing idea",
        turn,
    )
    sided = _POSITIONAL_LABEL.sub(
        lambda match: (
            "this idea"
            if _POSITIONAL_SLOTS[(match.group(1) or match.group(2)).lower()] == mine
            else "the opposing idea"
        ),
        sided,
    )
    sided = _GLOSSED_LABEL.sub("", sided)
    # The label was a proper noun and opened sentences as one, so the substitution
    # leaves a lower-case word where the sentence's capital was.
    return _SENTENCE_OPENER.sub(lambda match: match.group(0).upper(), sided)


# A debater quoting the goal ends the sentence inside the quotation -- ...scavenging
# mechanisms." Hypothesis 1 is perfectly tailored... -- so the terminator is not the
# character before the space and the capital was not restored.
#
# A colon is not one of these terminators. It reads like one to a pattern, and treating
# it as one printed "the strongest argument is this: This idea concedes the point" --
# a capital in the middle of a sentence, which is exactly the fault this pattern exists
# to repair, introduced by the repair. "Hypothesis 1" was a proper noun and kept its
# capital after a colon; "this idea" is not and does not.
_SENTENCE_OPENER = re.compile(
    r"(?:(?<=^)|(?<=[.!?]\s)|(?<=[.!?][”\"')]\s)|(?<=\*\*\s))t(?=h(?:is|e) )"
)


def _idea_matches(
    record: ResearchRecord, candidate_id: str
) -> tuple[list[IdeaMatch], int, int, int]:
    matches: list[IdeaMatch] = []
    wins = losses = ties = 0
    for comparison in record.tournament.comparisons if record.tournament else []:
        if candidate_id not in {comparison.candidate_a_id, comparison.candidate_b_id}:
            continue
        opponent = (
            comparison.candidate_b_id
            if comparison.candidate_a_id == candidate_id
            else comparison.candidate_a_id
        )
        if comparison.winner_id == candidate_id:
            outcome = "win"
            wins += 1
        elif comparison.winner_id is None:
            outcome = "draw"
            ties += 1
        else:
            outcome = "loss"
            losses += 1
        matches.append(
            IdeaMatch(
                round_number=comparison.round_number,
                opponent_title=record.title_for(opponent),
                outcome=outcome,
                elo_before=comparison.elo_before.get(candidate_id, 0.0),
                elo_after=comparison.elo_after.get(candidate_id, 0.0),
                confidence=comparison.confidence,
                rationale=_sided(
                    _sentence(
                        comparison.rationale,
                        fallback=(
                            "The judge recorded no readable rationale for this match."
                        ),
                    ),
                    first=comparison.presented_first_id == candidate_id,
                ),
                judge=comparison.judge,
                # A turn that arrived as a serialised payload is dropped rather than
                # printed; the renderer reports how many were unreadable instead.
                debate_turns=[
                    _sided(turn, first=comparison.presented_first_id == candidate_id)
                    for turn in (
                        _sentence(item, fallback="") for item in comparison.debate_turns
                    )
                    if turn
                ],
                unreadable_turns=sum(
                    1
                    for item in comparison.debate_turns
                    if not _sentence(item, fallback="")
                ),
            )
        )
    # Chain the printed ratings so each row opens where the one above it closed.
    running = round(matches[0].elo_before) if matches else 0
    chained: list[IdeaMatch] = []
    for match in matches:
        chained.append(replace(match, shown_before=running))
        running = chained[-1].shown_after
    return chained, wins, losses, ties


def _idea_description(facts: dict[str, str], alternatives: Sequence[str]) -> list[str]:
    """The idea stated at length, before any reviewer has been allowed an opinion.

    Each paragraph used to close on a sentence saying what the field it had just
    printed was for: what a mechanism carries, why a prediction has to discriminate,
    what stating a falsifier in advance buys. Each is true of every idea, and printed
    under all eight of them they came to thirty-two paragraphs of glossary interleaved
    with the only text a reader is here for. The glossary is stated once in the
    preamble above the deep dives; what is left here is the idea.
    """
    # Two of the four lead-ins were bare noun phrases closed with a full stop --
    # "The predictions that separate it from its neighbours." and "What would falsify
    # it." -- while the third was a sentence, so one block of four paragraphs mixed
    # fragments with sentences. In the PDF and the DOCX there is no visual cue that
    # these are labels, and a full stop after a noun phrase reads as a truncation.
    #
    # The third was also a hedge: "It competes with at least one other reading of the
    # same situation." stood above however many readings the specialist recorded, and
    # on a live run that was two under every one of the eight ideas. The count is on
    # the record, so the sentence states it and runs into the readings themselves.
    count = len(alternatives)
    competing = (
        f"{_opening(count, 'competing reading', 'competing readings')} of the same "
        "situation " + ("has" if count == 1 else "have") + " to be displaced: "
        f"{_spliced(facts['Alternative explanations'])}."
        if count and _stated(facts, "Alternative explanations")
        else ""
    )
    return [
        facts["Mechanism and rationale"],
        (
            "These are the predictions that separate it from its neighbours: "
            f"{_spliced(facts['Discriminating predictions'])}."
            if _stated(facts, "Discriminating predictions")
            else facts["Discriminating predictions"]
        ),
        (competing if competing else facts["Alternative explanations"]),
        (
            f"This is the result that would falsify it: {_spliced(facts['Falsifier'])}."
            if _stated(facts, "Falsifier")
            else "No falsifier was recorded for this idea, so nothing here states "
            "what result would count against it."
        ),
    ]


def build_idea_briefs(record: ResearchRecord) -> list[IdeaBrief]:
    """Assemble one deep-dive brief per candidate, ordered by tournament rank."""
    ratings = record.tournament.ratings if record.tournament else {}
    shortlist = set(record.tournament.shortlist_ids) if record.tournament else set()
    ordered = sorted(
        record.candidates,
        key=lambda candidate: (-ratings.get(candidate.id, 0.0), candidate.id),
    )
    # The rating each idea is reported at is where its own match table ends, not a
    # separate rounding of the stored figure. Rank order still follows the stored
    # ratings, which are what the tournament actually compared.
    played = {
        candidate.id: _idea_matches(record, candidate.id) for candidate in ordered
    }
    shown_elo = {
        candidate_id: (
            matches[-1].shown_after
            if matches
            else round(ratings.get(candidate_id, 1500.0))
        )
        for candidate_id, (matches, *_) in played.items()
    }
    reviewed = {
        candidate.id: _idea_reviews(record, candidate.id) for candidate in ordered
    }
    # A novelty score means nothing on its own -- five of five in a field that tops out
    # at three is not the same statement as five of five where half the ideas also
    # scored five. The field is only knowable across candidates, so it is gathered here.
    novelty_field = [
        review.score
        for reviews in reviewed.values()
        for review in reviews
        if review.section == "Novelty"
    ]
    briefs = []
    for rank, candidate in enumerate(ordered, start=1):
        matches, wins, losses, ties = played[candidate.id]
        citations = record.evidence_support.get(candidate.id)
        facts = _idea_facts(candidate)
        reviews = reviewed[candidate.id]
        elo = shown_elo[candidate.id]
        shortlisted = candidate.id in shortlist
        accepted_flaw = record.override_for(candidate.id)
        verification_lead_in, verification = _deep_verification(reviews, facts)
        coherence, coherence_notes = _coherence(reviews, facts)
        # Three ideas finished a live tournament on the same Elo and the shortlist cut
        # fell inside that block: two were carried forward and one held back on an
        # identical rating and an identical mean review score. The verdict line stated
        # the outcome as a decision. Whether the tie straddles the cut is knowable
        # here and nowhere below, so it is worked out here and handed down.
        tied_ids = [
            other.id
            for other in ordered
            if other.id != candidate.id and shown_elo[other.id] == elo
        ]
        recommended = candidate.id in {
            record.ranked_id(item)
            for item in (
                record.manifest.recommendation_candidate_ids if record.manifest else []
            )
        }
        revised_lead_in, revised_form, revised_unchanged = _revised_form(
            record, candidate, recommended=recommended
        )
        briefs.append(
            IdeaBrief(
                title=record.title_for(candidate.id),
                candidate_id=candidate.id,
                rank=rank,
                elo=elo,
                category=_category_path(record, candidate),
                proposal=facts["Core idea"],
                description=_idea_description(facts, candidate.alternatives),
                facts=facts,
                summary=_summary_sections(
                    facts,
                    reviews,
                    rank=rank,
                    elo=elo,
                    shortlisted=shortlisted,
                    accepted_flaw=accepted_flaw,
                    tied_with=len(tied_ids),
                    tie_straddles_cut=any(
                        (other in shortlist) != shortlisted for other in tied_ids
                    ),
                    supporting=_supporting_claims(record, candidate),
                    novelty_field=novelty_field,
                ),
                table_rows=_table_rows(candidate),
                reviews=reviews,
                coherence=coherence,
                coherence_notes=coherence_notes,
                deep_verification=verification,
                deep_verification_lead_in=verification_lead_in,
                matches=matches,
                wins=wins,
                losses=losses,
                ties=ties,
                shortlisted=shortlisted,
                support=citations.support if citations else "unknown",
                unresolved_evidence_ids=list(citations.unresolved) if citations else [],
                accepted_flaw=accepted_flaw,
                tied_with=len(tied_ids),
                contradicting_claims=_contradicting_claims(record, candidate),
                predictions=[_sentence(item) for item in candidate.predictions],
                alternatives=list(candidate.alternatives),
                revised_lead_in=revised_lead_in,
                revised_form=revised_form,
                revised_unchanged=revised_unchanged,
                strategy=candidate.generation_strategy.replace("_", " "),
                mermaid=candidate.workflow_diagram_mermaid.strip(),
                evidence_notes=_evidence_notes(record, candidate),
                revised_is_recommended=recommended,
            )
        )
    return briefs


@dataclass
class _EvidenceStatement:
    """A discovery finding with the leads that produced it, ready to be cited."""

    text: str
    urls: list[str]
    facet: str
    relation: str
    scope: list[str] = field(default_factory=list)
    """What discovery recorded as bounding this finding, verbatim.

    Every claim in both live runs carried one -- "specific to NCM811 cathodes and dry
    vs wet coating methods", "tested at elevated temperature (55 degrees C) to
    accelerate capacity fading" -- and the report printed none of them. A retention
    figure whose stated scope is one cathode chemistry, set down as a bare finding
    under Main Research Directions, is read as a general result about coatings.
    """


def _contradicting_claims(record: ResearchRecord, candidate: Candidate) -> list[str]:
    """The cited claims the evidence stage recorded as arguing against this idea."""
    cited = set(candidate.evidence_ids)
    return [
        _sentence(claim.claim)
        for claim in (record.evidence.claims if record.evidence else [])
        if claim.id in cited and claim.relation == "contradicts"
    ]


def _supporting_claims(record: ResearchRecord, candidate: Candidate) -> list[str]:
    """The cited claims the evidence stage recorded as arguing for this idea.

    The section headed "Supporting Arguments & Evidence" carried no evidence and no
    argument specific to the idea it sat under: two variants of one sentence covered
    all eight ideas, and the findings each idea was actually built on were printed
    only in the knowledge base, where nothing says which idea cites which. Each is
    numbered against the same reference list as the rest of the report.
    """
    cited = set(candidate.evidence_ids)
    sources = {
        item.id: item for item in (record.evidence.sources if record.evidence else [])
    }
    stated: list[str] = []
    for claim in record.evidence.claims if record.evidence else []:
        if claim.id not in cited or claim.relation != "supports":
            continue
        source = sources.get(claim.source_id or "")
        # Unannotated: a qualifier here would be counted against the ceiling that
        # keeps the tags rare, and the standing of this idea's evidence as a whole is
        # stated by the support verdict in the listing above.
        marker = record.citations.marker([source.url], annotate=False) if source else ""
        stated.append(
            _sentence(f"{_sentence(claim.claim).rstrip('.')} {marker}".strip())
        )
    return stated


_LOCATOR = re.compile(r"(?:https?://\S+|\b10\.\d{4,9}/\S+|\bPMID:?\s*\d+)", re.I)

VERIFIED_BADGE = "[Verified Source]"
LEAD_BADGE = "[Literature Lead]"
UNSOURCED_BADGE = "[Unsourced claim]"


@dataclass(frozen=True)
class _EvidenceRecord:
    """One record of this run's evidence base, as an evidence statement can name it."""

    identifier: str
    text: str
    status: str


# Shaped like an identifier and nothing like a sentence: one word, no spaces, at
# least one underscore-separated part. A statement that is only this is a
# specialist naming a record instead of saying what the record holds.
_BARE_REFERENCE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")


def _evidence_index(record: ResearchRecord) -> dict[str, _EvidenceRecord]:
    """Every record id a specialist could have cited, to what it says and its standing.

    The generation stage is given the evidence base with its identifiers and is
    asked for statements. On a live run four of the eight ideas answered with the
    identifiers: "Evidence for: claim_6_1." was printed to the reader as the whole
    of what the literature says for the second-ranked hypothesis, over an id
    defined nowhere in the document. Resolving them here prints the claim; it also
    gives the badge something to read, which is the other half of the same defect
    -- the badge was decided by looking for a URL in the sentence, so a statement
    citing a verified claim by id was labelled an unsourced one.
    """
    index: dict[str, _EvidenceRecord] = {}
    for source in record.evidence.sources if record.evidence else []:
        index[source.id] = _EvidenceRecord(
            source.id,
            # Whatever a statement resolves an id into is printed as the finding,
            # so the search's own furniture is cut here for the same reason it is
            # cut from the reference list.
            _without_search_chrome(" ".join(source.title.split())) or source.url,
            source.verification_status,
        )
    for lead in record.discovery.source_leads if record.discovery else []:
        index[lead.id] = _EvidenceRecord(
            lead.id,
            _without_search_chrome(" ".join(lead.title.split())) or lead.canonical_url,
            lead.verification_status,
        )
    for narrative in record.discovery.narratives if record.discovery else []:
        for statement in narrative.statements:
            index[statement.id] = _EvidenceRecord(
                statement.id, statement.text, "discovered_unverified"
            )
    # Claims last: where a claim and a source carry the same id, what the claim
    # says is closer to what the statement was citing it for.
    for claim in record.evidence.claims if record.evidence else []:
        index[claim.id] = _EvidenceRecord(
            claim.id, claim.claim, claim.verification_status
        )
    return index


def _cited_records(
    statement: str, index: Mapping[str, _EvidenceRecord]
) -> list[_EvidenceRecord]:
    """Which of the run's own records this statement names, in the order named."""
    found: list[_EvidenceRecord] = []
    for match in _BARE_REFERENCE.finditer(statement):
        entry = index.get(match.group(0))
        if entry is not None and entry not in found:
            found.append(entry)
    return found


def _grounding_badge(
    statement: str,
    verified: set[str],
    known: set[str],
    cited: Sequence[_EvidenceRecord] = (),
) -> str:
    """What stands behind one of a candidate's own evidence statements.

    Three labels rather than two, because the third case is the common one and
    the other two do not cover it: a specialist writes "contradictory findings
    are reported in the literature" and names no paper. Calling that a
    literature lead would tell a reader there is something to follow.

    A cited record outranks a URL in the prose, because it is the run's own
    verdict on the document rather than a guess from the text of a sentence.
    """
    if any(entry.status in GROUNDED_STATUSES for entry in cited):
        return VERIFIED_BADGE
    if cited:
        return LEAD_BADGE
    locators = {
        match.group(0).rstrip(".,;)").lower() for match in _LOCATOR.finditer(statement)
    }
    if not locators:
        return UNSOURCED_BADGE
    if any(any(locator in url for url in verified) for locator in locators):
        return VERIFIED_BADGE
    if any(any(locator in url for url in known) for locator in locators):
        return LEAD_BADGE
    # A locator the run never retrieved is a claim about a document, not a
    # record of one.
    return UNSOURCED_BADGE


def _evidence_notes(
    record: ResearchRecord, candidate: Candidate
) -> list[tuple[str, str, str]]:
    """The candidate's own for/against/missing statements, each labelled by grounding."""
    sources = record.evidence.sources if record.evidence else []
    known = {source.url.lower() for source in sources if source.url}
    verified = {
        source.url.lower()
        for source in sources
        if source.url and source.verification_status in GROUNDED_STATUSES
    }
    index = _evidence_index(record)
    names = _record_names(record)
    # As recorded, before the naming pass rewrote the ids out of them. Falling back
    # to the candidate's own fields covers a brief built from a record this function
    # was handed directly, which the tests do and the pipeline does not.
    recorded = record.cited_evidence.get(
        candidate.id,
        [
            list(candidate.evidence_for),
            list(candidate.evidence_against),
            list(candidate.evidence_gaps),
        ],
    )
    notes: list[tuple[str, str, str]] = []
    for heading, statements in zip(
        ("Evidence for", "Evidence against", "Evidence gaps"), recorded, strict=True
    ):
        for statement in statements:
            text = _sentence(statement)
            if not text:
                continue
            # A gap is a statement that no evidence exists, so grounding it is
            # not a question that can be asked of it.
            badge = (
                ""
                if heading == "Evidence gaps"
                else _grounding_badge(
                    text, verified, known, _cited_records(text, index)
                )
            )
            notes.append((heading, badge, _stated_evidence(text, index, names)))
    return notes


# "pass4_stmt_5: Increased coating thickness causes severe mass transfer resistance"
# -- the specialist citing the record it is about to quote, in the shape a footnote
# marker would take if this format had footnotes. The badge beside the bullet already
# says what stands behind it, so the marker is dropped and the sentence kept.
_CITED_PREFIX = re.compile(
    rf"^[\[(]?({_BARE_REFERENCE.pattern})[\])]?\s*[:\-\u2013\u2014]\s+"
)


def _stated_evidence(
    text: str, index: Mapping[str, _EvidenceRecord], names: Mapping[str, str]
) -> str:
    """One evidence statement with its ids read out, however the specialist wrote it."""
    bare = text.rstrip(".").strip()
    if _BARE_REFERENCE.fullmatch(bare):
        # The whole answer was an id. What the record says is the only thing here
        # worth printing; the id itself named nothing a reader of this page can look
        # up, and printing it beside the text was two ways of saying the same record.
        entry = index.get(bare)
        if entry is not None:
            return _sentence(entry.text)
        return _sentence(
            f'The specialist gave the record id "{bare}" here and no statement '
            "beside it, and no record of that id exists in this run's evidence base"
        )
    prefix = _CITED_PREFIX.match(text)
    if prefix and prefix.group(1) in index:
        text = _sentence(text[prefix.end() :])
    return _sentence(
        _BARE_REFERENCE.sub(
            lambda match: names.get(match.group(0), match.group(0)), text
        )
    )


def _shared_qualifications(record: ResearchRecord) -> list[str]:
    """The qualifications recorded against every claim, which are about the run.

    Discovery writes a scope against each finding it extracts, and the verification
    pass appends its own. On both live runs the verification pass overwrote the
    scopes with two sentences of process boilerplate -- that the claim came from a
    search snippet, and that the pass restated it without confirming it -- so every
    finding carried the identical pair. Printed against each of them that is the
    same paragraph six times over; what it says once is that nothing was checked.
    """
    claims = record.evidence.claims if record.evidence else []
    if len(claims) < 2:
        return []
    common = set(claims[0].limitations).intersection(
        *(set(claim.limitations) for claim in claims[1:])
    )
    # In recorded order, and in the order of the first claim, so the sentence does
    # not reshuffle itself between runs of the same report.
    return [item for item in claims[0].limitations if item in common]


def _evidence_statements(record: ResearchRecord) -> list[_EvidenceStatement]:
    shared = _shared_qualifications(record)
    statements: list[_EvidenceStatement] = []
    for narrative in record.discovery.narratives if record.discovery else []:
        for statement in narrative.statements:
            statements.append(
                _EvidenceStatement(
                    # Empty rather than the placeholder: a finding with no text is
                    # not a finding, and six of the fifty-five Main Research
                    # Directions on a live run read "Not stated by the specialist."
                    text=_sentence(statement.text, fallback=""),
                    urls=list(statement.source_urls),
                    facet=statement.facet,
                    relation=statement.relation,
                )
            )
    for claim in record.evidence.claims if record.evidence else []:
        source = next(
            (
                item
                for item in (record.evidence.sources if record.evidence else [])
                if item.id == claim.source_id
            ),
            None,
        )
        statements.append(
            _EvidenceStatement(
                text=_sentence(claim.claim, fallback=""),
                urls=[source.url] if source else [],
                facet="verified"
                if claim.verification_status == "verified"
                else "claim",
                relation=claim.relation,
                scope=[item for item in claim.limitations if item not in shared],
            )
        )
    return _merged_statements(statements)


def _merged_statements(
    statements: Sequence[_EvidenceStatement],
) -> list[_EvidenceStatement]:
    """One entry per finding, however many passes and claims recorded it.

    Seven Deep Research passes cover overlapping ground and the evidence stage then
    records the verified ones again as claims, so the same sentence arrives many
    times. Nothing deduplicated them: Main Research Directions printed 55 findings on
    a run that held 23, several of them three and four times over -- and because the
    relation is recorded per copy, two adjacent paragraphs stated the same finding
    and said that discovery had read it as supporting the hypothesis and as arguing
    against it. Six more were the empty-field placeholder, printed as findings.

    Merging keeps every locator and every recorded scope, prefers the verified copy,
    and where the copies disagree about which way the finding cuts, says that instead
    of picking one.
    """
    merged: dict[str, _EvidenceStatement] = {}
    relations: dict[str, list[str]] = {}
    for statement in statements:
        if not statement.text:
            continue
        key = _comparable(statement.text)
        relations.setdefault(key, []).append(statement.relation)
        held = merged.get(key)
        if held is None:
            merged[key] = replace(statement, urls=list(statement.urls))
            continue
        held.urls.extend(url for url in statement.urls if url not in held.urls)
        held.scope.extend(item for item in statement.scope if item not in held.scope)
        if statement.facet == "verified":
            held.facet = "verified"
    for key, statement in merged.items():
        recorded = set(relations[key])
        if len(recorded) > 1:
            statement.relation = _DISPUTED_RELATION
    return list(merged.values())


# How discovery recorded each finding as bearing on the question. "supports" carries
# no clause because it is the reading a cited finding gets by default; the other two
# have to displace that reading, so they say so.
_DISPUTED_RELATION = "recorded_both_ways"
"""Set on a finding whose copies did not agree about which way it cuts."""

_RELATION_CLAUSES = {
    "contradicts": "Discovery recorded this finding as arguing against the hypothesis "
    "the question puts, not for it.",
    "neutral": "Discovery recorded this finding as bearing on the question without "
    "arguing either way, so it is context rather than support.",
    _DISPUTED_RELATION: "Discovery returned this finding more than once and read it "
    "differently each time, so nothing on the record says which way it cuts.",
}

_RELATION_LEAD_INS = {
    "contradicts": "Discovery recorded the next {count} as arguing against the "
    "hypothesis the question puts, not for it.",
    "neutral": "Discovery recorded the next {count} as bearing on the question "
    "without arguing either way, so they are context rather than support.",
    _DISPUTED_RELATION: "Discovery returned each of the next {count} more than once "
    "and read it differently each time, so nothing on the record says which way "
    "they cut.",
}


def _grouped_by_relation(
    statements: Sequence[_EvidenceStatement],
) -> list[tuple[str, list[_EvidenceStatement]]]:
    """The findings gathered under each way discovery read them, default first.

    The relation clause is a property of the group and not of the finding, and
    printed under each finding it was the same sentence eleven times running on a
    live run -- with the four neutral ones and the four contradicting ones doing the
    same thing further down the section. Gathering them lets the clause be said once,
    and gives a reader looking for the case against the goal somewhere to look.
    """
    order = ["", *_RELATION_CLAUSES]
    groups: dict[str, list[_EvidenceStatement]] = {key: [] for key in order}
    for statement in statements:
        relation = statement.relation if statement.relation in _RELATION_CLAUSES else ""
        groups[relation].append(statement)
    return [(key, groups[key]) for key in order if groups[key]]


def _relation_lead_in(relation: str, count: int) -> str:
    """The clause a group of findings shares, stated once over the group."""
    return (
        _RELATION_LEAD_INS[relation].format(count=_plural(count, "finding"))
        + " That is said here once rather than repeated under each of them."
    )


def _cited(
    record: ResearchRecord,
    statement: _EvidenceStatement,
    hoisted: frozenset[str] = frozenset(),
) -> str:
    marker = record.citations.marker(statement.urls)
    text = statement.text if not marker else f"{statement.text.rstrip('.')} {marker}."
    # Which way a finding cuts is the one thing about it a reader cannot recover from
    # the sentence and its number. It used to be carried by a "(disputed)" tag on the
    # source, which said the source was doubtful rather than that the finding argued
    # the other way -- and the finding in question was neither doubted nor doubtful.
    #
    # Only "contradicts" was worded, and "neutral" fell through to the bare sentence,
    # which is the same output a supporting finding gets. Four of the twelve findings
    # on the live runs were neutral, printed in a section a reader opens looking for
    # the case for the goal, so a finding discovery had marked as neither for nor
    # against was read as one more piece of support.
    #
    # Where the findings that share a relation are printed together, the clause is
    # made over the group instead and ``hoisted`` names it, so the group's members
    # carry no copy of it.
    clause = (
        ""
        if statement.relation in hoisted
        else _RELATION_CLAUSES.get(statement.relation, "")
    )
    # The qualification discovery wrote against the finding travels with the finding.
    # Held back to the appendix, or dropped as these were, a result recorded as
    # holding for one chemistry at one temperature is read here as a general one.
    scope = (
        f"Discovery recorded {_plural(len(statement.scope), 'qualification')} on it: "
        f"{_join(statement.scope, fallback='none.')}"
        if statement.scope
        else ""
    )
    return " ".join(item for item in (text, clause, scope) if item)


@dataclass
class _Draft:
    """A section split into what it must say and what it may say if space allows."""

    number: int
    title: str
    core: list[str]
    extra: list[str] = field(default_factory=list)
    subsections: list[NarrativeSubsection] = field(default_factory=list)
    grids: list[NarrativeGrid] = field(default_factory=list)


_NO_EVIDENCE = (
    "No source lead survived discovery for this run, so nothing in this section "
    "carries a citation; every statement below is derived from the workflow's own "
    "artifacts rather than from external literature."
)


def _goal_title(session: Session) -> str:
    """A short document title, since the raw question runs to a paragraph."""
    return derive_idea_title(session.question, max_words=12)


def _for_the_goal(session: Session) -> str:
    """A heading's trailing "for <goal>", dropped where the goal is a question.

    Goals in this pipeline are almost always written as questions, and a question
    cannot be the object of a preposition: "Candidate Ideas for Can a Protective
    Interphase Coating Extend Lithium-ion Battery Cycle Life?" changes mood halfway
    through and ends on punctuation the heading did not ask for. The document's title
    is that question, set one page above, so the qualifier adds nothing there either.
    """
    title = _goal_title(session)
    return "" if title.rstrip().endswith("?") else f" for {title}"


# A count on a scale is a figure; a count of things is a word. "A spread of 2 points"
# is an arithmetic result and reads as one, but "raised against at least 7 of the 8
# ideas" is prose about the field and reads as a table cell that escaped. The samples
# this report imitates keep the split: minutes, hours and equivalents in numerals,
# clusters, cases and intermediates in words.
_MEASURED = frozenset({"point", "minute", "hour", "day", "week", "month"})


def _counted(count: int, singular: str, plural: str | None = None) -> str:
    """A count in figures, for a fact list rather than a sentence.

    The run-facts block is a table set as bullets, and the split above is wrong inside
    one: its tournament line prints a distribution beside a total, so spelling only the
    leading count gave "four rounds of 4, 4, 4, and 6 matches, 18 in all" -- a clause
    that disagrees with itself about how it writes the same kind of number.
    """
    return f"{count} {singular if count == 1 else (plural or _plural_of(singular))}"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if singular in _MEASURED or count >= len(_NUMBER_WORDS):
        return _counted(count, singular, plural)
    noun = _counted(count, singular, plural).partition(" ")[2]
    return f"{_number_word(count).lower()} {noun}"


# Prose does not open a sentence with a digit. "3 objections recurred across the
# field" and "1 claim this idea cites" both reached a live report, and a sentence
# starting in a numeral reads as a list item that lost its bullet.
_NUMBER_WORDS = (
    "No One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve".split()
)
_LOWER_NUMBER_WORDS = frozenset(word.lower() for word in _NUMBER_WORDS)


_TEEN_WORDS = "Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen".split()
_TENS_WORDS = "Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()


def _number_word(count: int) -> str:
    """A number as a word, capitalised for the start of a sentence.

    The table stopped at twelve, which is where house style stops spelling counts out
    in running prose. But a count that opens a sentence has to be a word whatever its
    size, and a live run's review-band paragraph opened "15 reviews closed at two or
    below" -- a numeral where the reader expects a capital. Spelling is only ever
    reached through this function; ``_plural`` still writes figures above twelve in
    the middle of a sentence, which is what house style asks for.
    """
    if count < len(_NUMBER_WORDS):
        return _NUMBER_WORDS[count]
    if count < 20:
        return _TEEN_WORDS[count - 13]
    if count < 100:
        tens, units = divmod(count, 10)
        word = _TENS_WORDS[tens - 2]
        return word if not units else f"{word}-{_NUMBER_WORDS[units].lower()}"
    if count < 1000:
        hundreds, rest = divmod(count, 100)
        word = f"{_NUMBER_WORDS[hundreds]} hundred"
        return word if not rest else f"{word} and {_number_word(rest).lower()}"
    return str(count)


def _opening(count: int, singular: str, plural: str | None = None) -> str:
    """A count that opens a sentence, spelled the way a sentence spells it."""
    counted = _plural(count, singular, plural)
    return f"{_number_word(count)} {counted.partition(' ')[2]}"


# The pipeline's stage ids are its own vocabulary, not the reader's: "reflect" is the
# review pass, "proximity" is the clustering pass, and neither word says so. Worse, a
# set of ids has no order, so the sorted list a report printed opened on "evidence" and
# closed on "scope" -- the last stage first and the first stage last.
_STAGE_WORDS = {
    "scope": "scoping the goal",
    "evidence": "literature discovery",
    "generate": "idea generation",
    "reflect": "independent review",
    "rank": "tournament ranking",
    "evolve": "evolution of the shortlist",
    "proximity": "clustering by mechanism",
    "meta_review": "meta-review",
    "report": "report assembly",
}


def _stage_words(stages: Iterable[str]) -> list[str]:
    """The named stages, in the order the pipeline runs them rather than in id order."""
    present = set(stages)
    named = [_STAGE_WORDS[stage] for stage in STAGES if stage in present]
    return named + sorted(
        stage.replace("_", " ") for stage in present if stage not in _STAGE_WORDS
    )


def _listed(items: Sequence[str], *, fallback: str = "none") -> str:
    """A comma list, for short phrases that carry no punctuation of their own.

    ``_join`` separates on semicolons because it folds whole stated sentences into one
    another. On a list of two-word stage names that punctuates for prose that is not
    there: "covering evidence; evolve; generate; and scope" reads as four clauses.
    """
    if not items:
        return fallback
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _plural_of(singular: str) -> str:
    """The regular plural, including the sibilant case that produced "18 matchs"."""
    if singular.endswith(("s", "x", "z", "ch", "sh")):
        return singular + "es"
    if singular.endswith("y") and singular[-2:-1] not in "aeiou":
        return singular[:-1] + "ies"
    return singular + "s"


def _section_one(record: ResearchRecord) -> _Draft:
    session, plan = record.session, record.plan
    mode = session.research_mode.replace("_", " ")
    # The goal the report is titled for and the question the run executed are two
    # different strings, and only the first was ever printed. On the live run the
    # second named the cathode surface, the eighty-per-cent retention threshold and
    # the uncoated control -- none of which the goal says -- and every idea below was
    # generated and reviewed against it. Withholding the operative question means the
    # reader judges the answers against a question that was never asked.
    restated = bool(
        plan
        and plan.question
        and _content_words(plan.question) != _content_words(session.question)
    )
    # Where both are printed the goal itself is on the cover under Goal, a page above,
    # and quoting it here as well put the same sentence in the document three times
    # counting the title. Where the run executed the goal as written there is no
    # second form of it, and this is the only place in the overview it appears.
    opening = (
        "This report addresses a single research goal, stated under Goal on the cover "
        "and restated below in the form the run executed."
        if restated
        else f"This report addresses a single research goal: {_sentence(session.question)}"
    )
    # The four sentences that used to close this paragraph -- nothing here is a
    # finding, a source is not verified until it is read, auto approval is not
    # approval -- are true of every run word for word, and putting them fourth in
    # the report taught a reader to skim the opening. They are in the standing-limits
    # advisory now, and the pointer below says how many advisories there are.
    core = [
        f"{opening} "
        f"The goal was declared as {mode} work with an intended claim of "
        f"{_sentence(plan.intended_claim if plan else 'hypothesis').rstrip('.')}, which "
        "fixes the standard of proof every idea below is held to."
    ]
    if restated:
        core.append(
            "Scoping restated that goal as the question the run went on to execute: "
            f"{_sentence(plan.question)} This is the narrower formulation, and it is "
            "the one the ideas below were generated, reviewed and ranked against. "
            "Where it and the goal on the cover differ, an idea that answers this "
            "question has not necessarily answered the goal."
        )
    if plan and plan.constraints:
        # The sentence here used to read "ideas that would breach a constraint were
        # not eligible for promotion regardless of how well they scored elsewhere",
        # which describes a gate this pipeline does not have. No stage screens a
        # candidate against the plan's constraints, and on a live run all eight ideas
        # were promoted without any of them being checked against one. Claiming an
        # enforcement that does not exist is worse than having none, because it tells
        # the reader not to perform the check themselves.
        core.append(
            # The four constraints used to be reprinted here in full, one numbered
            # clause each, three hundred words after the cover printed the same four
            # under Requirements. Numbering them on the cover instead gives this
            # paragraph something to point at and leaves it free to say the thing the
            # cover cannot: that nothing checked them.
            "The goal carries "
            + _number_word(len(plan.constraints)).lower()
            + " explicit constraints that bound what may legitimately be proposed. "
            "They are set out and numbered under Requirements on the cover, and those "
            "numbers are what the rest of this report refers to. No stage of this run "
            "screened the ideas against them mechanically, and compliance was not a "
            "condition of being ranked or promoted. "
            + _constraint_coverage(record, plan.constraints)
        )
    if plan and plan.assumptions:
        core.append(
            # Likewise listed on the cover, under Attributes. What this paragraph adds
            # is their standing, which is the part a reader acts on.
            "The plan also records "
            + _number_word(len(plan.assumptions)).lower()
            + " assumptions the work rests on, listed under Attributes on the cover. "
            "They matter because an idea inherits every assumption its goal was framed "
            "with. Each is an inference rather than an observation, none was tested in "
            "this run, and a reviewer who disagrees with one should expect the "
            "corresponding ideas to weaken."
        )
    # A governance block that a person answered is a decision about this report's
    # contents, so it belongs beside the approval regime rather than in an appendix.
    if record.adjudications:
        withdrawn, overridden = len(record.withdrawals), len(record.overrides)
        parts = []
        if withdrawn:
            parts.append(
                f"{_plural(withdrawn, 'hypothesis', 'hypotheses')} "
                f"{'was' if withdrawn == 1 else 'were'} withdrawn from the population "
                "and did not compete"
            )
        if overridden:
            parts.append(
                f"{_plural(overridden, 'hypothesis', 'hypotheses')} "
                f"{'is' if overridden == 1 else 'are'} still live while carrying a "
                "fatal flaw that was accepted by hand rather than resolved"
            )
        answered = {item.candidate_id for item in record.adjudications}
        # The flaws nobody ruled on. The paragraph reported the adjudicated ones and
        # stopped, so a run in which a person answered one flaw and 16 others were left
        # standing read as a run whose fatal flaws had all been dealt with -- and the
        # sentence said "a fatal flaw against one hypothesis ... and each one was
        # answered", disagreeing with itself about how many there were.
        unanswered = record.recorded_fatal_flaw_ids - answered
        core.append(
            "The safety and governance review recorded a fatal flaw against "
            f"{_plural(len(record.adjudications), 'hypothesis', 'hypotheses')} in this "
            "run, and "
            + ("it was" if len(record.adjudications) == 1 else "each of them was")
            # "Answered by a named human rather than by the workflow" was asserted of
            # whatever string the adjudication carried. On the run that finished today
            # all three were answered under the name "Automated verification run
            # (Claude Code)", and the report told its reader a human had decided them.
            # The name is free text on the command and nothing authenticates it, so
            # what the record establishes is that a decision was taken outside the
            # review -- and the reader is given the name to judge for themselves.
            + " answered outside the review, under "
            + (
                "the name "
                if len({item.adjudicator for item in record.adjudications}) == 1
                else "the names "
            )
            + _joined_titles(
                sorted({item.adjudicator for item in record.adjudications})
            ).rstrip(".")
            + ". Nothing in this system authenticates "
            + (
                "that name, so it records"
                if len({item.adjudicator for item in record.adjudications}) == 1
                else "those names, so they record"
            )
            + " what was claimed at the time and not who is accountable. "
            f"{_join(parts, fallback='No resolution was recorded.')} Every "
            "decision is reprinted in full, with its flaw and its written "
            "justification, under Governance adjudications below; a safety decision "
            "summarised is a safety decision a reader cannot check."
            + (
                ""
                if not unanswered
                # "That is the flaw a person ruled on." used to precede this, and it
                # said a third time what the paragraph opens by saying. Landing after
                # the pointer to Governance adjudications, its "that" reached for the
                # nearest noun and found the reprinting rather than the flaw. The
                # contrast it was there to draw is carried by "the other reviews".
                else " The other reviews recorded fatal flaws against a further "
                f"{_plural(len(unanswered), 'idea')}, which nobody adjudicated: those "
                "stand as written, under the ideas that carry them."
            )
        )
    blocking = [item for item in session.input_requirements if not item.resolved]
    if blocking:
        core.append(
            "Input sufficiency was checked before any idea was generated. "
            + _join(
                [
                    f"{item.input_type} is {item.status} because {item.reason.rstrip('.')}"
                    for item in blocking
                ],
                fallback="No input requirement was recorded.",
            )
            + " Because a required input is absent, the workflow was restricted to what "
            "the literature can support and no residue-level or dataset-level claim "
            "may be read into the ideas that follow."
        )
    if plan and plan.stopping_criteria:
        # Core, not elaboration. As an optional paragraph the word budget dropped it
        # from every live report, and one of the criteria it dropped was the rule that
        # stops the work on a thermal runaway. A stopping rule that is only printed
        # when there happens to be room is not a stopping rule.
        core.append(
            "Stopping criteria were declared up front so the run could not drift. "
            f"{_join(plan.stopping_criteria, fallback='None recorded.')} These are the "
            "conditions under which further work would have added cost without adding "
            "decision value, or must not continue at all. Nothing in this workflow "
            "enforces them; they bind whoever takes the work forward."
        )
    # The governance obligations used to be stated here as well as under
    # Recommendations and Next Steps, in two paragraphs that both listed them in full
    # and both sat in the droppable queue -- so the report either said it twice or,
    # more often, not at all. They are stated once, where the work they gate is
    # proposed, and pointed at from here.
    extra = []
    return _Draft(1, "Research Goal", core, extra)


def _labelled_bullets(items: Sequence[str]) -> str:
    """``Label: value`` items as a list rather than as one semicolon chain.

    Seven criteria run together came out as a two-hundred-word sentence whose sixth
    clause a reader has to count semicolons to find, and the fold to lower case that
    ``_join`` applies to everything after the first item demoted six labels that the
    rest of the report prints capitalised. They are a list; they are set as one.
    """
    bullets = []
    for item in items:
        text = " ".join(item.split())
        label, separator, rest = text.partition(":")
        if separator and rest.strip() and len(label.split()) <= 4:
            bullets.append(f"- **{label.strip()}** — {rest.strip().rstrip('.')}.")
        else:
            bullets.append(f"- {_sentence(text)}")
    return "\n".join(bullets)


def _section_two(record: ResearchRecord) -> _Draft:
    criteria = record.population.comparison_criteria if record.population else []
    plan = record.plan
    # Whether a judge ever read those criteria. A run whose tournament fell back to
    # arithmetic ranks on the review scores and never sees them, and the section opened
    # by telling that reader "every idea in this report was assessed against the seven
    # comparison criteria on the cover, applied identically to all of them so that the
    # ranking reflects the ideas" -- a description of the run that was configured, not
    # of the run that happened.
    judged = record.judged_by_model
    applied = (
        "applied identically to all of them so that the ranking reflects the ideas "
        "rather than the order in which they were written."
        if judged
        else "settled before the ideas were written. The tournament did not read "
        "them: every match in it was decided by arithmetic on the review scores "
        "rather than by a judge, so the criteria bear on the ranking below only "
        "through whatever the reviewers made of them."
    )
    core = [
        # The seven criteria are printed on the cover under Criteria, and printing
        # them again here put the same seven labels twice in the first two pages of
        # the document. What this section is for is what the cover cannot say: that
        # the set was fixed before the ideas were written and applied to all of them.
        "Every idea in this report was assessed against "
        + (
            f"the {_number_word(len(criteria)).lower()} comparison criteria set out on "
            f"the cover, {applied}"
            if criteria
            else f"one fixed set of criteria, {applied} No cross-candidate criterion "
            "was recorded."
        )
    ]
    core.append(
        "Five independent reviews were run against each idea: evidence and "
        "correctness, novelty, methods and feasibility, impact, and safety and "
        "governance. Safety is read separately from impact because it decides whether "
        "the work may proceed at all rather than whether it is worth proceeding. Each "
        # The legend used to read the scale straight off the recommendation --
        # five advance, three revise, two insufficient, one reject -- and then
        # mention the confidence adjustment as an afterthought. The adjustment is
        # not an afterthought: every reviewer in a live run states a confidence at
        # or above 0.80, so every printed score is the adjusted one and the legend
        # named the wrong verdict for each of them.
        "review answers one question on a five-point scale. The recommendation sets "
        "the number — advance is five, revise it first is three, evidence too thin "
        "to judge on is two, reject it is one — and the reviewer's own stated "
        "confidence then moves it one point up at 0.80 or above and one point down "
        "below 0.30. A printed four is therefore a confidently held revise and a "
        "printed two a confidently held rejection, so the number carries the verdict "
        "and the conviction behind it together and neither can be read off it alone. "
        "A reviewer who records a "
        "fatal flaw caps the score at two whatever the recommendation, because an "
        "unresolved fatal flaw is disqualifying rather than merely costly."
    )
    if plan and plan.success_criteria:
        core.append(
            "Success for the goal as a whole was defined separately from per-idea "
            "scoring, and is stated under Criteria on the cover, in its own block "
            "apart from the comparison criteria the scoring used. No stage of this "
            # "Nothing in this run measured an idea against it" was a claim about the
            # whole run resting on the absence of a stage: reviewers and judges write
            # about the goal constantly, and one of them may well have weighed an idea
            # against a success criterion in prose. What the record supports is the
            # narrower statement -- no score on this page is a score against them --
            # and the old closing clause then made the opposite promise, that the
            # ranking exposes the gap. Nothing in the run measures it, so it cannot.
            "run scored an idea against it. An idea can therefore score well on the "
            "comparison criteria and still fail to advance the goal, and no ranking "
            "below closes that gap: it has to be closed by reading the success "
            "criteria against whichever idea is being considered."
        )
    extra = [
        "Scores are decision aids and not evidence. They compress a reviewer's "
        "judgement into one number so that ideas can be ordered, but the ordering is "
        "only as good as the evidence underneath it; where the evidence is thin the "
        "scores below cluster tightly, and a tight cluster should be read as "
        "'undifferentiated' rather than as 'equally good'."
    ]
    if record.tournament:
        extra.append(
            "Ranking itself used pairwise comparison rather than absolute scoring. "
            f"The tournament ran {_plural(record.tournament.swiss_rounds, 'Swiss round')} "
            f"followed by a round robin over the top {record.tournament.top_round_robin_size}, "
            "updating an Elo rating after every match. Pairwise comparison was chosen "
            "because reviewers are far more reliable at saying which of two ideas is "
            "stronger than at placing a single idea on an absolute scale."
        )
    return _Draft(2, "Evaluation Criteria", core, extra)


# Who each recorded actor is, in words. The ids are internal, and "cli researcher"
# with the underscore taken out is still an internal id with a space in it.
_ACTOR_WORDS = {
    "cli_researcher": "a researcher working through the command line",
    "researcher": "a researcher",
    "supervisor": "the run supervisor",
    "auto_approval_policy": "the approval profile, with no person involved",
    "operator": "an operator",
}


def _actor_words(actor: str) -> str:
    """An actor named as a person or a policy, never as the id the log holds."""
    return _ACTOR_WORDS.get(actor, f"the actor recorded as {actor.replace('_', ' ')}")


def _named_mechanisms(clusters: Sequence[ResearchCluster]) -> bool:
    """Whether the recorded mechanisms tell the clusters apart.

    A fallback clustering writes one sentence into every cluster -- four of them read
    "candidates share a generation lens but retain distinct predictions" -- and the
    report then promised "each named here with the mechanism its members share" above
    four copies of it, told the reader that two ideas in one cluster fail for the same
    reason, and sent them to Main Research Directions for a mechanism stated nowhere.
    """
    mechanisms = {
        _spliced(cluster.shared_mechanism).lower()
        for cluster in clusters
        if cluster.shared_mechanism.strip()
    }
    return len(mechanisms) == len(clusters)


def _research_directions(record: ResearchRecord) -> list[str]:
    """The directions discovery drew from the literature, each said once.

    Deduplicated, and with the goal itself struck out. Each pass reports its own
    directions and the passes overlap, so the list ran to one entry per pass with
    repeats -- and a pass that fell back to its raw report records the goal question
    as its direction, which put "Does a protective coating improve rechargeable
    battery cycle life?" at the top of Research Directions twice, above the
    directions actually drawn from the literature.
    """
    goal = _comparable(record.session.question)
    return list(
        dict.fromkeys(
            said
            for narrative in (record.discovery.narratives if record.discovery else [])
            for direction in narrative.research_directions
            if (said := _sentence(direction)) and _comparable(said) != goal
        )
    )


def _section_three(record: ResearchRecord) -> _Draft:
    statements = _evidence_statements(record)
    # Counted off the same list the reader is pointed at. Counted off the raw field,
    # this paragraph promised more directions than Research directions printed.
    directions = _research_directions(record)
    core = []
    if directions:
        core.append(
            # The directions are printed as a list under Research directions, a page
            # below, and were spelled out here as well: three multi-word names read
            # twice in one overview. Naming them once and pointing at the list keeps
            # this paragraph doing the thing the list cannot, which is saying what
            # kind of thing they are.
            "Discovery identified the directions along which the literature is "
            "currently moving, and these framed what the generator was asked to "
            f"produce: {_number_word(len(directions)).lower()} of them, listed under "
            "Research directions below. Each is a region of the problem rather than a "
            "proposal in itself."
        )
    elif statements:
        # Saying nothing came from the literature, immediately above four cited
        # findings, was the report contradicting itself. A search-grounded pass
        # returns findings without the synthesis Deep Research writes over them,
        # which is a thinner result than a direction set, not an absent one.
        #
        # Which pass ran is the reason for the thinness, and the report used to hold
        # it back to an appendix two thousand lines below. Read here, "discovery
        # searched the literature and did not synthesise it" is a finding about the
        # field; read with the substitution, it is a fact about the configuration
        # this run was executed under, and only the second is true.
        core.append(
            "Discovery searched the literature but did not synthesise it into a set "
            "of research directions, so what follows is individual findings rather "
            "than a map of where the field is moving. "
            + (
                "That is a property of the pass that ran rather than of the "
                "literature: the Deep Research agent, which iterates its searches "
                "until coverage stops improving and writes the synthesis, did not "
                "run on this goal, and a single search-grounded pass stood in for "
                "it. What that substitution cost is set out under Literature "
                "discovery in the provenance section at the end of this report. "
                if record.deep_research_stood_in
                else ""
            )
            + "The idea space below was framed by these findings and by the question "
            "itself, and a reader should not read the list as a survey of what has "
            "already been tried."
        )
    else:
        core.append(
            "Discovery did not return a set of literature-derived research directions "
            "for this goal, so the directions below were inferred from the structure "
            "of the goal itself rather than from published work. That is a material "
            "weakness: it means the idea space was bounded by the framing of the "
            "question and not by what the field has already tried."
        )
    if statements:
        shared = _shared_qualifications(record)
        if shared:
            # The qualifications come back as whole sentences, one of them with a
            # colon of its own, so _join sets them as sentences and not as a series.
            # Introduced by a colon they then read as a single item that full stops
            # have broken up -- "below: Unverified claim inferred from search snippet
            # and title. Restored to the discovered wording: ..." opens a list and
            # closes nothing. The lead-in ends on a full stop instead, and says the
            # words are the run's rather than the report's.
            #
            # "That is a statement" also stood over two of them.
            one = len(shared) == 1
            core.append(
                f"The same {_number_word(len(shared)).lower()} "
                f"{_plural(len(shared), 'qualification').partition(' ')[2]} "
                + ("is" if one else "are")
                + " recorded against every finding below, in the words the run "
                + ("recorded it" if one else "recorded them")
                + ". "
                + _join(shared, fallback="none.")
                + (
                    " That is a statement about how this run reached the findings "
                    "rather than about any one of them, so it is made here once and "
                    "not repeated under each."
                    if one
                    else " Those are statements about how this run reached the "
                    "findings rather than about any one of them, so they are made "
                    "here once and not repeated under each."
                )
            )
        for relation, group in _grouped_by_relation(statements):
            # A group of one is the finding's own sentence either way, and a lead-in
            # over it would be a second paragraph saying what the first says.
            hoisted = frozenset({relation} if relation and len(group) > 1 else ())
            if hoisted:
                core.append(_relation_lead_in(relation, len(group)))
            head = group if relation else group[:4]
            core.extend(_cited(record, statement, hoisted) for statement in head)
            # The tail used to be optional elaboration, and the reference list is
            # built from the citation markers the report actually emits. So when the
            # word budget dropped that paragraph, the one source cited only there
            # vanished from the references -- while the deep dives went on discussing
            # it by title, leaving a source named four times in the document and
            # listed nowhere. The findings are introduced by one sentence rather than
            # dropped, which bounds nothing but says why the list runs on.
            if not relation and group[4:]:
                # One paragraph each, as above. Folded into a single sentence they ran
                # to a 350-word block in which six findings, their qualifications and
                # their relations to the question were separated by nothing a reader
                # could see, and folding saved no words -- the length is the same.
                one = len(group[4:]) == 1
                core.append(
                    f"Discovery returned {_plural(len(group[4:]), 'further finding')}, "
                    + ("which carries " if one else "which carry ")
                    + "the same standing as those above and "
                    + ("is" if one else "are")
                    + " stated here so that nothing the report cites is missing from "
                    "its references."
                )
                core.extend(
                    _cited(record, statement, hoisted) for statement in group[4:]
                )
    else:
        core.append(_NO_EVIDENCE)
    clusters = record.landscape.clusters if record.landscape else []
    if clusters:
        distinct = _named_mechanisms(clusters)
        core.append(
            # This is the one place the shared mechanisms are spelled out. They used to
            # be printed again as the research-direction bullets and a third time under
            # every converging pair, so a mechanism a reader had already met arrived
            # twice more in the same document, each time as though it were new.
            #
            # The five of them also used to be a single semicolon chain inside one
            # sentence -- two hundred words between the colon and the full stop, with
            # each cluster's name, size and mechanism separated from the next by a
            # semicolon and from its own parts by commas. A sentence per cluster says
            # the same thing and can be read.
            "Mapping the generated ideas back onto the problem produced "
            + (
                f"{_plural(len(clusters), 'distinct cluster')}, each named here with "
                "the mechanism its members share. "
                if distinct
                else f"{_plural(len(clusters), 'cluster')}. The clustering stage "
                "recorded no mechanism that tells them apart, so the names below "
                "group the ideas without saying what the grouping rests on. "
            )
            + " ".join(
                f"{cluster.name} holds "
                + _plural(len(cluster.candidate_ids), "idea")
                + (
                    f" around {_spliced(cluster.shared_mechanism)}."
                    if distinct
                    else "."
                )
                for cluster in clusters
            )
            + (
                " Clustering matters for the recommendation: two ideas in the same "
                "cluster fail for the same reason, so funding both buys less "
                "information than the pair of scores would suggest."
                if distinct
                else " Clustering is meant to show where funding two ideas buys one "
                "idea's worth of information, and that reading is not available here: "
                "a shared name is not a shared failure mode."
            )
        )
        # The clustering stage may put one idea under two mechanisms, and nothing
        # said so. A live report opened "three distinct clusters" and then gave
        # their sizes as two, two and one over four ideas -- and printed the same
        # hypothesis under two different converging pairs further down, each pair
        # described as the two ideas resting on a single shared mechanism.
        repeated = [
            candidate_id
            for candidate_id, count in Counter(
                candidate_id
                for cluster in clusters
                for candidate_id in cluster.candidate_ids
            ).items()
            if count > 1
        ]
        if repeated:
            core.append(
                _opening(len(repeated), "idea appears", "ideas appear")
                + " in more than one cluster: "
                + _joined_titles(
                    [record.title_for(item) for item in repeated], fallback="none"
                )
                + ". The sizes above count "
                + ("it" if len(repeated) == 1 else "each of them")
                + " once per cluster, so they total more than the number of ideas "
                "mapped, and a result against any of the mechanisms "
                + ("it was" if len(repeated) == 1 else "they were")
                + " placed under reaches "
                + ("it" if len(repeated) == 1 else "them")
                + "."
            )
    extra = []
    if record.landscape and record.landscape.coverage_gaps:
        extra.append(
            "The map also shows where nothing was proposed. "
            f"{_join(record.landscape.coverage_gaps, fallback='No gap was recorded.')} A "
            "coverage gap is not evidence that the region is unpromising; it is "
            "evidence that this run did not look there."
        )
    return _Draft(3, "Main Research Directions", core, extra)


def _shortlist_prerequisites(brief: IdeaBrief) -> str:
    """What has to be in place before an idea starts, and what would stop it.

    The three fields this draws on are each optional and each a list, so the sentence
    that introduces them cannot count them or assume they are there. It used to do
    both: "turns on a stated threshold" over four thresholds, and "one set of inputs
    has to exist: no external dependency was recorded for this idea".
    """
    facts = brief.facts
    parts = [
        f"Before {brief.title} can be attempted at all, its inputs have to exist: "
        f"{_spliced(facts['Required inputs and dependencies'])}."
        if _stated(facts, "Required inputs and dependencies")
        else f"Nothing was recorded that {brief.title} depends on, so the run does "
        "not say what has to be in place before it can start."
    ]
    if _stated(facts, "Go/no-go tests"):
        parts.append(
            "Whether to continue with it or abandon it is decided against what the "
            f"specialist set down in advance: {_spliced(facts['Go/no-go tests'])}. "
            "Setting that down before the work starts is what keeps the idea "
            "falsifiable rather than merely plausible."
        )
    else:
        parts.append(
            "No go/no-go threshold was set down for it either, so there is no "
            "stated result that would end the work rather than extend it."
        )
    if _stated(facts, "Principal risks"):
        parts.append(
            f"Against that sits what pursuing it would risk: "
            f"{_spliced(facts['Principal risks'])}."
        )
    return " ".join(parts)


def _shared_contradiction_notice(
    briefs: Sequence[IdeaBrief], shared: Sequence[str]
) -> str:
    """A finding several ideas cite against themselves, stated once with those ideas.

    What the reader needs from it does not change between chapters -- the finding, and
    that a case resting on it has to answer it -- and naming the ideas together is
    something no one chapter can do.
    """
    one = len(shared) == 1
    lead = (
        f"{_opening(len(shared), 'finding')} cited below "
        + ("was" if one else "were")
        + " recorded by the evidence stage as cutting against the research question "
        "rather than for it, and more than one idea rests part of its case on "
        + ("it" if one else "them")
        + ". "
        + ("It is" if one else "They are")
        + " stated here, with the ideas that cite "
        + ("it" if one else "them")
        + ", rather than under each of those ideas in turn. An idea citing "
        + ("it" if one else "one of them")
        + " has to account for the finding rather than pass over it."
    )
    stated = [
        f"{_sentence(claim)} Cited by "
        + _joined_titles(
            [brief.title for brief in briefs if claim in brief.contradicting_claims],
            fallback="no idea",
        )
        + "."
        for claim in shared
    ]
    return " ".join([lead, *stated])


def _section_four(record: ResearchRecord, briefs: Sequence[IdeaBrief]) -> _Draft:
    """The candidate ideas, one numbered subsection and one comparison grid each."""
    title = f"Candidate Ideas{_for_the_goal(record.session)}"
    withdrawals = record.withdrawals
    generated = len(briefs) + len(withdrawals)
    core = [
        f"The generator produced {_plural(generated, 'idea')} across the strategies "
        "described above, each stated as a claim that can be shown to be wrong. They "
        "are set out here in rank order, each with the same five-row grid so that they "
        "can be read against one another. The grid carries the idea in the "
        "specialist's own words: the mechanism it rests on, the prediction that "
        "separates it from its neighbours, the test that would show it wrong, the "
        # Serial comma: the other three items are comma-separated and only the last
        # pair was not, so the fifth row read as part of the fourth.
        # "beside" for a paragraph that is set above the grid, not next to it.
        "dependency it turns on, and its principal risk. The prose above each grid "
        "does not repeat those; the reviews each idea received and the matches it "
        "played are in its own section later in this report."
    ]
    grounding, grounding_hoisted = shared_support_notices(
        [brief.support for brief in briefs]
    )
    if grounding:
        core.append(grounding)
    # The same contradicting finding was cited by three of the eight ideas on a live
    # run, and each of those chapters printed the finding in full under the same two
    # sentences of explanation: ninety identical words, three times, inside one
    # section. Stated once over the ideas that share it, each chapter says only that
    # it is one of them.
    cited_against = Counter(
        claim for brief in briefs for claim in brief.contradicting_claims
    )
    shared_against = [claim for claim in cited_against if cited_against[claim] > 1]
    if shared_against:
        core.append(_shared_contradiction_notice(briefs, shared_against))
    if withdrawals:
        # Renumbering around a withdrawn idea would leave the reader counting seven
        # where eight were written, which reads as a smaller run rather than a cut one.
        core.append(
            f"{_opening(len(withdrawals), 'of those ideas is', 'of those ideas are')} "
            "not ranked below. "
            + _joined_titles(
                [item.title for item in withdrawals], fallback="No idea was withdrawn"
            )
            + f" {'was' if len(withdrawals) == 1 else 'were'} withdrawn after the "
            "safety and governance review recorded a fatal flaw, so "
            f"{'it' if len(withdrawals) == 1 else 'they'} never entered the "
            f"tournament. {'It is' if len(withdrawals) == 1 else 'They are'} kept in "
            "the numbering below, unranked, with the decision that removed "
            f"{'it' if len(withdrawals) == 1 else 'them'} and the reason given."
        )
    subsections = []
    for index, brief in enumerate(briefs, start=1):
        paragraphs = []
        # An accepted fatal flaw outranks even a broken citation: the idea is live,
        # someone knew it was dangerous, and the reader must meet that before the claim.
        if brief.governance_notice:
            paragraphs.append(brief.governance_notice)
        # An idea whose grounding does not exist has to say so before it says
        # anything else; a reader must never meet the claim first.
        if brief.support_is_alarming:
            paragraphs.append(brief.support_notice)
        # Only the claim. The paragraph used to restate the rationale and the falsifier
        # in full, and the grid four lines below printed both again word for word --
        # eight ideas, sixteen duplications, and a reader who cannot tell whether the
        # second copy differs from the first without comparing them clause by clause.
        # What the grid holds is now said once, in the lead-in above.
        paragraphs.append(brief.facts["Core idea"])
        # Beyond the top ten the second paragraph is dropped rather than the idea: a
        # long field would otherwise push the narrative through its word ceiling.
        if index <= 10:
            # "The reading it has to displace is that A; and B" put one reading in the
            # subject, two in the predicate, and an "is that" that does not distribute
            # over the semicolon between them. The count is on the brief, so the
            # sentence can agree with it.
            competing = brief.alternatives
            if not competing:
                displace = (
                    "No competing reading was recorded against it, so nothing here "
                    "says what a positive result would have to rule out."
                )
            elif len(competing) == 1:
                displace = (
                    "The reading it has to displace is that "
                    f"{_spliced(brief.facts['Alternative explanations'])}."
                )
            else:
                displace = (
                    f"It has {_plural(len(competing), 'competing reading')} to "
                    f"displace: {_spliced(brief.facts['Alternative explanations'])}."
                )
            # The grid states the first prediction, so the prose states the rest. A
            # list that opened on the grid's own sentence read as though the report
            # had lost track of what it had just printed.
            further = brief.predictions[1:]
            # "the grid" without "below" is a backward reference to a table the reader
            # has not reached: this paragraph is printed above it.
            #
            # The subject is a pronoun rather than the title. Two lines under a heading
            # that is the title, "Beyond the prediction in the grid below, A Self-
            # healing Polyurethane-based Interphase Coating Containing Microencapsulated
            # Lithium Salts is separated from its competitors" is a twenty-word subject
            # repeating the heading verbatim, and every other sentence in the paragraph
            # already says "It".
            predicts = (
                "Beyond the prediction in the grid below, it is separated from "
                f"its competitors by {_plural(len(further), 'further prediction')}: "
                # Spliced so the list opens in lower case after the colon, and stopped
                # again afterwards: splicing is for folding a sentence into a longer
                # one, so it takes the full stop with it.
                f"{_sentence(_spliced(_join(further, fallback='none.')))}"
                if further
                else "It rests on the single prediction in the grid below and "
                "records no other."
            )
            # A rank the tournament did not produce is not reported as one. Three live
            # ideas held an Elo of 1184 and were printed here as ranks four, five and
            # six, each in its own subsection and each reading as a result -- what put
            # them in that order was the sort's tie-break. Section five discloses the
            # tie, four pages from the reader who is reading one idea.
            standing = (
                f"It finished level with "
                f"{_plural(brief.tied_with, 'other idea')} on an Elo of {brief.elo}, "
                f"listed at rank {brief.rank} by sort order."
                if brief.tied_with
                else f"It finished rank {brief.rank} on an Elo of {brief.elo}."
            )
            # The verdict belongs beside the idea it is a verdict on; what it means
            # does not change from idea to idea, and printed in full here it was the
            # same three lines under five of the eight. Where it recurs the paragraph
            # carries the verdict alone and the explanation is in the lead above.
            verdict = (
                ""
                if brief.support_is_alarming
                else f"Its grounding is marked {brief.support_label}."
                if brief.support in grounding_hoisted
                else brief.support_prose
            )
            paragraphs.append(f"{predicts} {displace} {standing} {verdict}".rstrip())
        # A citation that cuts against the idea carrying it is the one thing about an
        # idea's grounding a reader cannot recover from a support verdict, because the
        # verdict counts citations rather than reading them.
        hoisted_against = [
            claim for claim in brief.contradicting_claims if claim in shared_against
        ]
        own_against = [
            claim for claim in brief.contradicting_claims if claim not in shared_against
        ]
        if hoisted_against:
            one = len(hoisted_against) == 1
            paragraphs.append(
                f"{_opening(len(hoisted_against), 'finding')} this idea cites "
                + ("was" if one else "were")
                + " recorded by the evidence stage as cutting against the research "
                "question rather than for it, and "
                + ("it is" if one else "they are")
                + " stated at the head of this section with the other ideas that cite "
                + ("it." if one else "them.")
            )
        if own_against:
            paragraphs.append(
                # The relation is to the research question, not to this idea, and the
                # paragraph used to say "contradicting it" with "it" reading as the
                # idea. On a live run the contradicting finding was that overly thick
                # coatings reduce performance, printed under an idea about dry versus
                # wet coating at 2 wt%, which it neither supports nor refutes.
                f"{_opening(len(own_against), 'claim')} this idea cites "
                + ("was" if len(own_against) == 1 else "were")
                + " recorded by the evidence stage as cutting against the research "
                "question rather than for it: "
                + _join(own_against, fallback="none.")
                # "The citation is genuine and the finding is real" was printed
                # unconditionally, two paragraphs under a support verdict that had
                # just said none of this idea's citations had been checked against
                # its source. The report cannot both call a finding real and say
                # nobody confirmed it, so the sentence now says only what the
                # resolution stage established and defers the rest to the verdict.
                + (
                    " The citation resolves to a record in this session"
                    + (
                        ", and that record has been checked against its source."
                        if brief.support in {"grounded", "partially_grounded"}
                        else ", though the record itself is unverified."
                    )
                )
                + " An idea that cites it is resting part of its case on a finding the "
                "evidence stage read the other way, which the case has to account for "
                "rather than pass over."
            )
        subsections.append(
            NarrativeSubsection(
                number=f"4.{index}",
                title=brief.title,
                paragraphs=paragraphs,
                table_rows=brief.table_rows,
            )
        )
    # Withdrawn ideas keep a slot at the end rather than a rank. A slot is what makes
    # the removal visible; a rank would claim it competed, which it never did.
    for offset, note in enumerate(withdrawals, start=len(briefs) + 1):
        paragraphs = [note.notice(adjudications_ahead=True)]
        # The notice directly above already says the idea never entered the tournament
        # and carries no rank and no Elo. Repeating that here put the same fact in two
        # consecutive paragraphs; what the notice does not say, and what a reader
        # looking for the idea further down needs, is that this slot is all there is.
        absent = (
            " This slot is the whole of its presence in the report; there is no "
            "section of its own further down."
        )
        if note.claim:
            paragraphs.append(
                f"The hypothesis as written was: {note.claim} It is reproduced here so "
                "that the decision to withdraw it can be judged against what was "
                "actually proposed." + absent
            )
        else:
            paragraphs.append(
                "The text of the withdrawn hypothesis is no longer recoverable from "
                "this session's artifacts, so only the decision and its reason can be "
                "shown." + absent
            )
        subsections.append(
            NarrativeSubsection(
                number=f"4.{offset}",
                title=f"{note.title} (withdrawn)",
                paragraphs=paragraphs,
            )
        )
    extra = [_shortlist_prerequisites(brief) for brief in briefs]
    return _Draft(4, title, core, extra, subsections)


def _section_five(record: ResearchRecord, briefs: Sequence[IdeaBrief]) -> _Draft:
    tournament = record.tournament
    core = []
    grids: list[NarrativeGrid] = []
    if tournament and tournament.comparisons:
        played = len(tournament.comparisons)
        decided = sum(1 for item in tournament.comparisons if item.winner_id)
        debated = sum(1 for item in tournament.comparisons if item.debate_turns)
        # "over 18 matches, of which 18 produced a winner" prints a count the reader
        # has to compare against the one three words earlier to learn that it says
        # nothing. What is worth stating is the exception, so only the drawn matches
        # are counted, and a clean sweep is said in words rather than in a repeat.
        drawn = played - decided
        core.append(
            f"The ideas were compared head to head over {_plural(played, 'match')}, "
            + (
                "every one of which produced a winner"
                if not drawn
                else f"of which {drawn} ended in a draw"
            )
            + ". Ratings started level and moved only "
            "on the outcome of a match, so an idea's final rating is a statement about "
            "who it beat rather than about how it read in isolation."
        )
        # What decided the undebated matches used to be stated as arithmetic from the
        # review scores, and the reader was told an arithmetic decision carries no
        # inspectable reasoning. On every live run those matches were judged by a
        # model that recorded a rationale, and the Judge column beside each of them
        # said so. The same wording had already been corrected in the per-idea
        # tournament block; this copy of it had not. It is read off the record now.
        undebated_judges = sorted(
            {
                _judge_label(item.judge)
                for item in tournament.comparisons
                if not item.debate_turns
            }
        )
        if debated and played - debated:
            core.append(
                f"{_number_word(debated)} of those matches "
                + ("was" if debated == 1 else "were")
                + " decided by a simulated scientific "
                "debate in which each side argued for its idea and a judge ruled; the "
                # "settled in a single pass by a single-pass model comparison": the
                # frame and the judge label were each carrying the same word, and the
                # sentence stuttered wherever the label was the common one.
                f"remaining {_number_word(played - debated).lower()} were decided by "
                f"{' and '.join(undebated_judges)}. The distinction matters when "
                "reading a close result: a single-pass verdict records the judge's "
                "reason but no exchange behind it, so it cannot be audited the way a "
                "debated match can. Both are reproduced in each idea's own section."
            )
        elif debated:
            core.append(
                "Every match was decided by a simulated scientific debate in which "
                "each side argued for its idea and a judge ruled, and the transcripts "
                "are reproduced in each idea's own section."
            )
        else:
            core.append(
                "Every match was decided by "
                f"{' and '.join(undebated_judges) or 'the ranking stage'} rather than "
                "by argument. No exchange underlies any individual result, so the "
                "ordering should be read as a summary of the reviews rather than as "
                "an independent judgement of the ideas."
            )
    else:
        core.append(
            "No pairwise tournament was recorded for this run, so the ideas below are "
            "presented in generation order and carry no comparative ranking."
        )
    if briefs:
        leader = briefs[0]
        core.append(
            f"{leader.title} finished first on {_plural(leader.wins, 'win')} against "
            f"{_plural(leader.losses, 'loss', 'losses')}, ending on an Elo of "
            f"{leader.elo}. " + _lead_over_rival(leader, briefs)
        )
        # Written out as a sentence this was ninety words of "A at 1290 (6-0); B at
        # 1234 (4-2); ..." -- a table read serially, in which comparing two rows means
        # holding both in mind. The position column shares a number between ideas that
        # finished level, because the tournament did not order them and the sort that
        # did is not a result.
        core.append(
            "The full standings follow. Ideas that finished level on rating share a "
            "position; the order among them below is the sort's and not a result."
        )
        ranks: dict[str, int] = {}
        for index, brief in enumerate(briefs, start=1):
            level = next(
                (
                    other
                    for other in briefs[: index - 1]
                    if other.elo == brief.elo and other.candidate_id in ranks
                ),
                None,
            )
            ranks[brief.candidate_id] = ranks[level.candidate_id] if level else index
        drew = any(brief.ties for brief in briefs)
        grids.append(
            NarrativeGrid(
                after=len(core) - 1,
                columns=[
                    "Position",
                    "Idea",
                    "Elo",
                    "Record (W-L-D)" if drew else "Record (W-L)",
                ],
                rows=[
                    [
                        str(ranks[brief.candidate_id]),
                        brief.title,
                        str(brief.elo),
                        f"{brief.wins}-{brief.losses}-{brief.ties}"
                        if drew
                        else f"{brief.wins}-{brief.losses}",
                    ]
                    for brief in briefs
                ],
            )
        )
    if tournament:
        # Core, not elaboration. Whether the ordering settled is the qualification on
        # every ranking claim above it, and as an optional paragraph it was the first
        # thing the word budget dropped -- leaving "the ordering of the top two is
        # real" standing over a tournament the record says never converged.
        core.append(_convergence(tournament, briefs))
    shortlisted = [brief for brief in briefs if brief.shortlisted]
    if shortlisted:
        # Core, not elaboration: the shortlist is what the next stage acted on, and
        # two things about how it was drawn are only visible from the standings.
        core.append(
            "The shortlist carried forward into evolution and re-review was "
            + _sentence(
                _joined_titles([brief.title for brief in shortlisted]),
                fallback="empty.",
            )
            + " Shortlisting is a budget decision rather than a verdict: an idea left "
            "out may still be correct, and one carried forward may still fail its "
            "first go/no-go test."
        )
        core.extend(_shortlist_caveats(shortlisted, briefs))
    extra = []
    for brief in briefs:
        if not brief.matches:
            continue
        decisive = max(brief.matches, key=lambda match: abs(match.swing))
        extra.append(
            f"The match that moved {brief.title} furthest was its round "
            f"{decisive.round_number} pairing against {decisive.opponent_title}, a "
            f"{decisive.outcome} that took it from {decisive.shown_before} to "
            f"{decisive.shown_after} at a stated confidence of {decisive.confidence:.2f}. "
            f"{decisive.rationale}"
        )
    return _Draft(5, "Comparison of Candidate Ideas", core, extra, grids=grids)


def _separating_criteria(separating: Sequence[str], level: Sequence[str]) -> str:
    """Which criteria a choice between the ideas may rest on, said once over all.

    The verdict belongs to the set of criteria and not to any one of them: what a
    reader wants from this section is which of the five are worth choosing on, and
    that answer was previously spread across five paragraphs in the same words.
    """
    named = (_listed(list(separating)), _listed(list(level)))
    opened = tuple(text[:1].upper() + text[1:] for text in named)
    if not separating:
        return (
            f"{opened[1]} left the ideas level or within a point of each other, so "
            "no criterion here separates the field and no choice between the ideas "
            "should be justified on these scores."
        )
    lead = (
        f"{opened[0]} spread the ideas by at least two points, which is wide enough "
        "to choose on."
    )
    if not level:
        return lead
    return (
        f"{lead} {opened[1]} did not, and a choice between the ideas should not be "
        "justified on " + ("it." if len(level) == 1 else "them.")
    )


def _section_six(record: ResearchRecord, briefs: Sequence[IdeaBrief]) -> _Draft:
    by_criterion: dict[str, list[tuple[str, int]]] = {}
    for brief in briefs:
        for review in brief.reviews:
            by_criterion.setdefault(review.section, []).append(
                (brief.title, review.score)
            )
    # The two the example is drawn on, taken from the criteria this run actually
    # scored. Written out as "evidence" and "feasibility", the sentence named a
    # criterion the section does not have: the paragraphs below it are headed
    # correctness, novelty, feasibility, impact and safety, so a reader who went
    # looking for the evidence paragraph the example promised found none.
    names = [name.lower() for name in by_criterion]
    example = (
        f"An idea that leads overall while trailing on {names[0]} is a different "
        f"proposition from one that leads on {names[0]} while trailing on "
        f"{names[-1]}, and the two call for different next steps."
        if len(names) > 1
        else ""
    )
    # Where the tournament had a judge, the ordering is a function of the matches and
    # these scores are not an input to it at all. Each criterion below used to close on
    # a spread being "wide enough to be doing real work in the final ordering", which
    # credited them with an influence the run gives them only when no judge was
    # available -- and said it once per criterion, three times over in one section.
    feeds = (
        "None of these scores enters the ordering directly: that comes from the "
        "matches in section five. What a spread here shows is whether the reviewers "
        "agreed, and on what."
        if record.judged_by_model
        else "The ordering in section five was computed from these scores, no judge "
        "having been available to compare the ideas, so a criterion that separates "
        "the field here is one that moved the ranking there."
    )
    core = [
        # Hard-coded as "four" against five criterion paragraphs printed directly
        # below it, in a report whose second section says five reviews were run.
        f"Ranking compresses {_plural(len(by_criterion), 'separate judgement')} "
        f"into one number, so this section unpacks them again. {example} "
        f"{feeds}".strip()
    ]
    # Which criteria separated the field, so the verdict can be given over all of
    # them at once. Printed under each paragraph it was the same closing sentence
    # four times in five on a live run -- "wide enough to separate the field", with
    # only the counts in front of it changing -- and a reader who wants to know which
    # criteria are worth choosing on had to collect the answer paragraph by paragraph.
    separating: list[str] = []
    level: list[str] = []
    for criterion, scored in by_criterion.items():
        top = max(score for _, score in scored)
        bottom = min(score for _, score in scored)
        average = sum(score for _, score in scored) / len(scored)
        at_top = [title for title, score in scored if score == top]
        at_bottom = [title for title, score in scored if score == bottom]
        (separating if top - bottom > 1 else level).append(criterion.lower())
        if top == bottom:
            # With every review on the same score there is no highest and no lowest,
            # and printing both produced "Seven of the ideas tied highest at 3, and 7
            # of the ideas tied lowest at 3" -- the same set named twice, as two
            # different things.
            body = f"Every review on this criterion came in at {top}."
        else:
            # Closing on a bare "the spread is wide enough to be doing real work"
            # printed the identical sentence under three of the five criteria, which
            # reads as a template rather than as a judgement. Saying how wide, and how
            # many ideas sit at the top of it, is what distinguishes one spread.
            #
            # What sits at the top of the spread is an idea. Calling it a review named
            # the instrument rather than the thing measured, two clauses after the same
            # set was introduced as "four of the ideas". A point on a five-point review
            # scale is a count of grades and not a measured quantity the way an Elo
            # point is, so it is spelled -- "a spread of 2 points with two ideas at the
            # top of it" wrote the same kind of number both ways inside one clause.
            body = (
                f"{_placed(at_top, top, 'highest', opening=True)}, and "
                f"{_placed(at_bottom, bottom, 'lowest')}. That is a spread of "
                f"{_number_word(top - bottom).lower()} "
                f"{'point' if top - bottom == 1 else 'points'} with "
                f"{_plural(len(at_top), 'idea')} at the top of it."
            )
        core.append(
            f"On {criterion.lower()}, the ideas averaged {average:.1f} out of five. "
            + body
        )
    if by_criterion:
        core.append(_separating_criteria(separating, level))
    # Recurrence is what makes this paragraph worth printing, so it is measured
    # rather than assumed. Sorting the distinct objections and taking the first four
    # printed four objections raised once each against one idea apiece, under a
    # sentence announcing that the same objections recur -- while the two objections
    # that were raised against every idea in the field went unmentioned.
    spread = _recurring_objections(briefs)
    if spread:
        core.append(_objection_spread(spread, len(briefs)))
    elif any(review.objections for brief in briefs for review in brief.reviews):
        core.append(
            "No objection was raised against more than one idea. The reviewers found "
            "different faults in each proposal rather than a fault running through the "
            "field, so the objections below are grounds for choosing between the ideas "
            "rather than grounds for regenerating them."
        )
    extra = []
    flaws = [
        (brief.title, finding)
        for brief in briefs
        for review in brief.reviews
        for finding in review.findings
        if review.score <= 2
    ]
    for title, finding in flaws[:6]:
        extra.append(
            # The closing gloss used to read that a score of two or below means the
            # reviewer could not judge the idea on the evidence available rather
            # than that the idea was judged and found wanting. Both halves are
            # wrong under the scale section 2 sets out: two is what a confidently
            # held rejection scores, and it is also the cap applied to any review
            # that records a fatal flaw. The report was talking a reviewer's
            # verdict down.
            f"A low-scoring review of {title} recorded the following. {finding} A "
            "score of two or below is a rejection, a finding that the evidence is "
            "too thin to judge on, or any review at all that recorded a fatal flaw; "
            "which of the three it is can be read off that review's own answer."
        )
    # Section five is "Comparison of Candidate Ideas" and this one was "Comparative
    # Analysis of Candidate Ideas": two headings a reader cannot tell apart, on facing
    # pages, over a tournament and a review-score breakdown respectively. The reference
    # reports keep the pair distinct -- "Idea Comparison and Synthesis" with
    # "Comparative Analysis" -- and the clone had taken the long form of both.
    return _Draft(6, "Comparative Analysis by Review Criterion", core, extra)


def _recurring_objections(briefs: Sequence[IdeaBrief]) -> list[tuple[str, int]]:
    """Objections raised against at least half the field, commonest first.

    Two reviewers writing the same objection about two ideas do not write the same
    sentence -- "No power rationale for N=10" and "No power rationale provided for
    N=12" are one objection -- so exact text cannot count recurrence. Objections are
    grouped by what they are about, with the numbers that differ per idea already
    dropped by the tokeniser, and the shortest phrasing in a group represents it.
    """
    groups: list[tuple[set[str], list[str], set[str]]] = []
    for brief in briefs:
        for review in brief.reviews:
            for objection in review.objections:
                topic = _objection_topic(objection)
                if len(topic) < 2:
                    continue
                match = next(
                    (group for group in groups if _same_topic(topic, group[0])), None
                )
                if match is None:
                    groups.append((topic, [objection], {brief.title}))
                else:
                    match[1].append(objection)
                    match[2].add(brief.title)
    # A phrasing without a figure in it first, and only then the shortest. The
    # shortest phrasing of the sample-size objection was "No power rationale for
    # N=10", which the paragraph then reported as the objection raised against seven
    # of eight ideas -- three of which propose N=15. A representative has to be true
    # of the group it represents, and a number carried over from one idea's protocol
    # is the one part of the wording that is not.
    recurring = [
        (
            min(phrasings, key=lambda text: (_carries_a_figure(text), len(text))),
            len(ideas),
        )
        for _, phrasings, ideas in groups
        if len(ideas) * 2 >= len(briefs) and len(ideas) > 1
    ]
    return sorted(recurring, key=lambda item: (-item[1], item[0]))[:3]


_A_FIGURE = re.compile(r"\d")


def _carries_a_figure(text: str) -> bool:
    """Whether a phrasing pins itself to one idea's numbers."""
    return bool(_A_FIGURE.search(text))


def _objection_spread(spread: Sequence[tuple[str, int]], field: int) -> str:
    """The objections that recur, and what a reader can do about them.

    "At least", because the objections are grouped by what they are about and two
    phrasings of one complaint can still miss each other. The count is a floor the
    record supports.
    """
    counts = {count for _, count in spread}
    size = _number_word(field).lower()
    # The count is hoisted where every objection carries the same one. Printed per
    # item it was the same eight-word clause three times in a sentence the reader
    # has to parse around three em dashes: "lacks randomization and blinding --
    # raised against at least seven of the eight ideas, needs a specified
    # statistical test for multiple comparisons -- raised against at least seven of
    # the eight ideas, and no power rationale for N=10 -- raised against at least
    # seven of the eight ideas". Hoisting out of a single item is not worth it: the
    # colon then introduces a one-item list and "each was raised" counts to one.
    shared = counts.pop() if len(spread) > 1 and len(counts) == 1 else 0
    items = [
        _sentence(objection).rstrip(".")
        + (
            ""
            if shared
            else f" — raised against at least {_number_word(count).lower()} "
            f"of the {size} ideas"
        )
        # Where every recorded phrasing of an objection names a figure, the
        # representative has to carry one, and the figure belongs to whichever
        # protocol it was raised against rather than to the field the sentence is
        # describing.
        + (" (quoted with one idea's figures)" if _carries_a_figure(objection) else "")
        for objection, count in spread
    ]
    # The closing gloss is written for the near-unanimous case, and _recurring_objections
    # thresholds at half the field. At four of seven it was false twice over: three ideas
    # do not carry the objection, so choosing one of those three resolves it exactly, and
    # an objection three ideas escape is not a property of the goal.
    escaped = field - min(count for _, count in spread)
    return (
        f"{_opening(len(spread), 'objection')} recurred across the field, which is a "
        "stronger signal than any single score. "
        + (
            f"Each was raised against at least {_number_word(shared).lower()} of the "
            f"{size} ideas: "
            if shared
            else ""
        )
        + _join(items, fallback="No objection was recorded.")
        + (
            " An objection raised against all but one of the ideas is a property of "
            "the goal rather than of the ideas, and cannot be resolved by choosing "
            "differently among them."
            if escaped <= 1
            else f" {_number_word(escaped)} of the ideas escaped "
            + (
                "it, so it is not a property of the goal: whether a choice inherits "
                "it depends on which idea is chosen."
                if len(spread) == 1
                else "at least one of these, so they are not properties of the goal: "
                "which of them a choice inherits depends on which idea is chosen."
            )
        )
    )


def _same_topic(one: set[str], other: set[str]) -> bool:
    """Whether two objections are about the same thing.

    Two shared topic words is the floor -- a single one merges "no power rationale"
    with "no statistical plan" on the word they happen to share -- and the shorter of
    the two has to be mostly covered, so a long objection cannot swallow a short one
    that only brushes it.
    """
    overlap = len(one & other)
    return overlap >= 2 and overlap / min(len(one), len(other)) >= 0.4


def _objection_topic(text: str) -> set[str]:
    """What an objection is about, with the phrasing and the per-idea numbers gone.

    Five reviewers wrote the same objection five ways -- "Lacks randomization and
    blinding", "Missing explicit randomization and blinding protocols", "Fails to
    explicitly state that cell assignment and testing will be randomized and
    blinded" -- so the complaint verb, the hedging and the inflection all have to
    come off before two of them can be recognised as one.
    """
    words = _content_words(text) - _OBJECTION_VERBS
    return {_stem(word) for word in words if not word.isdigit()}


_OBJECTION_VERBS = frozenset(
    """lacks fails failed fail does do explicit explicitly state states stated
    mention mentions provide provides provided require requires required needs
    specify specifies specified formal proposal propose proposes currently
    given without still simply also""".split()
)


_SUFFIXES = (
    "ization",
    "isation",
    "ations",
    "ation",
    "ments",
    "ment",
    "ings",
    "ing",
    "ed",
    "es",
    "s",
    # Only ever reached on a stem another suffix has already exposed -- "randomized"
    # loses "ed" to leave "randomiz". No English word ends in a bare "iz", so this
    # cannot over-merge two words that were different to begin with.
    "iz",
)


def _stem(word: str) -> str:
    """Enough of a stem to match "randomization" to "randomized".

    Stripped repeatedly rather than once: "randomized" loses its "ed" and stops at
    "randomiz", which shares no token with the "random" that "randomization" reduces
    to, so the two phrasings of one objection stayed in separate groups and the
    recurrence count came out one idea short. Four characters have to survive, which
    is what keeps "basis" and "size" whole.
    """
    for _ in range(len(_SUFFIXES)):
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        else:
            break
    return word


def _placed(
    titles: Sequence[str], score: int, end: str, *, opening: bool = False
) -> str:
    """Who sat at one end of a criterion, without naming one of several.

    ``max`` returns a single item whether or not the maximum is unique, and naming
    that item read as a finding: a live report said one idea "scored highest at 5" on
    impact where seven of eight scored 5, then admitted two clauses later that seven
    reviews sat at the top. A tie is a fact about the criterion, not about the idea
    the sort happened to reach first.
    """
    if len(titles) == 1:
        return f"{titles[0]} scored {end} at {score}"
    if len(titles) <= 3:
        return f"{_joined_titles(titles)} tied {end} at {score}"
    # Capitalised only where the clause opens a sentence: "4 of the ideas tied
    # highest" is a sentence starting on a digit, and "and Four of the ideas tied
    # lowest" is a number capitalised in the middle of one.
    count = _number_word(len(titles))
    return f"{count if opening else count.lower()} of the ideas tied {end} at {score}"


def _section_seven(record: ResearchRecord, briefs: Sequence[IdeaBrief]) -> _Draft:
    statements = _evidence_statements(record)
    novelty = [
        (brief.title, review)
        for brief in briefs
        for review in brief.reviews
        if review.section == "Novelty"
    ]
    core = [
        "An idea is only worth pursuing if it is not already settled, so each one was "
        "reviewed against what the field is understood to do today. That comparison is "
        "only as good as the prior art the workflow could actually see, and its limits "
        "are stated below rather than left implicit."
    ]
    if statements:
        core.append(
            # The first three statements used to be reprinted here word for word, a
            # few hundred words after Main Research Directions had printed the same
            # three and then two more. A section that compares against the literature
            # does have to say what it is comparing against -- but saying where it is
            # does that, and the reader has just read it.
            "The comparison is against the literature this run retrieved: the "
            f"{_plural(len(statements), 'finding')} stated in full under Main "
            "Research Directions above. The standing of those findings is the "
            "standing of every novelty judgement below."
        )
    else:
        core.append(
            "No external prior art was retrieved for this run. The novelty judgements "
            "below therefore compare each idea against the other ideas in this report "
            "and against general methodological practice, not against the published "
            "record. A claim of novelty on that basis is weak and should not be "
            "carried into a funding or publication decision."
        )
    if novelty:
        advancing = [title for title, review in novelty if review.score >= 4]
        marginal = [title for title, review in novelty if review.score <= 2]
        qualified = [
            title
            for title, review in novelty
            if review.score >= 4 and (review.objections or review.fatal_flaws)
        ]
        if advancing:
            core.append(
                "The novelty reviews judged the following to add something beyond "
                "current practice: "
                + _sentence(_joined_titles(advancing))
                # This closed by classifying every one of those increments as
                # methodological rather than empirical. Nothing in the run makes
                # that distinction -- a novelty review records a score, findings
                # and objections -- and the reason offered, that no result has been
                # produced, is true of every run of this workflow, so the clause
                # fired unconditionally and would have said the same of an idea
                # whose novelty was entirely empirical.
                + " These are the novelty reviewers' own judgements, made against "
                "the retrieved literature listed above rather than against a "
                "systematic prior-art search, and each review is printed in full "
                "under its idea."
                # A high novelty score and a prior-art objection from the same
                # review are both on the record, and this section printed only the
                # first. One live report placed an idea in this group on a five out
                # of five whose own reviewer had written "some literature already
                # discusses HF scavenging by amphoteric oxides like Al2O3 and ZnO"
                # -- the reservation was three hundred lines below, under the idea,
                # where nobody reading the comparison section would meet it.
                + (
                    " The verdict is not unqualified: the same review recorded an "
                    "objection against "
                    + (
                        "every one of them"
                        if len(qualified) == len(advancing)
                        else _joined_titles(qualified)
                    )
                    + ". A four or a five is the reviewer's answer to how much is "
                    "new, not a record that nothing was raised, and what was "
                    "raised is printed under the idea it was raised against."
                    if qualified
                    else ""
                )
            )
        if marginal:
            core.append(
                "The following were judged too close to established practice, or too "
                "poorly evidenced to place: "
                + _sentence_of_titles(marginal, fallback="none.")
                + " An idea in this group is not necessarily wrong; it is unlikely to "
                "return information the field does not already hold."
            )
    extra = []
    for title, review in novelty[:4]:
        if review.findings:
            extra.append(
                f"The novelty review of {title} recorded the following. "
                + " ".join(review.findings)
                + f" It closed at {review.score} out of five."
            )
    return _Draft(7, "Comparison with Existing Solutions", core, extra)


# What an open fatal flaw actually stops, read off the review that recorded it. A
# safety finding stops the work; a correctness finding stops the claim; a novelty or
# impact finding says the work is not worth doing, which is a different sentence
# altogether. All three used to print "has to be closed before any work proceeds",
# and on the live run the only flaw in this branch was a novelty review's -- the idea
# duplicates published work -- carrying stop-work language a safety reviewer had not
# written. Escalating a value judgement into a prohibition spends the credibility the
# report needs for the case where the prohibition is real.
_FLAW_CONSEQUENCES = {
    "Safety": "no work on it may begin until a named owner has cleared it",
    "Correctness": "the claim it attacks does not stand as written, so the flaw is "
    "prior to any protocol built on that claim",
    "Feasibility": "the design cannot be executed as specified, so the protocol has "
    "to be rewritten before it can be costed",
    "Novelty": "what it decides is whether the work is worth doing rather than "
    "whether it may be done",
    "Impact": "what it decides is whether the work is worth doing rather than whether "
    "it may be done",
}


def _open_flaw_consequence(record: ResearchRecord, ids: Sequence[str]) -> str:
    """What the open flaws stop, in the terms of the reviews that recorded them."""
    sections = sorted({item for id_ in ids for item in record.flaw_sections(id_)})
    stated = [
        f"the {section.lower()} review's, which means {_FLAW_CONSEQUENCES[section]}"
        for section in sections
        if section in _FLAW_CONSEQUENCES
    ]
    if not stated:
        return (
            "Nothing in this run tested it, so it has to be settled before the idea "
            "is worked on."
        )
    return (
        ("The flaw is " if len(stated) == 1 else "The flaws are ")
        + _series(stated)
        + ". Nothing in this run tested "
        + ("it" if len(stated) == 1 else "them")
        + "."
    )


def _reference_standing(record: ResearchRecord) -> str:
    """Whether the sources this report cites were read, said in one sentence.

    This used to be the flat assertion that every one of them was a lead nobody had
    checked, printed on every run whatever verification had established. On the run
    that finished today it stood two hundred lines below "every claim any of them
    cites exists in this report and has been verified against its source", and the
    two sentences described the same sources.
    """
    checked, total = record.citations.verification_standing
    outstanding = (
        "the workflow recorded where a statement came from, but inspecting the "
        "original and confirming that it says what is attributed to it remains "
        "outstanding"
    )
    if not total or not checked:
        return f"Every one of them is a lead rather than a verified reference: {outstanding}."
    if checked == total:
        return (
            "Every one of them was retrieved and checked against the document it "
            "names, so a marker in the text points at a source this run has read."
        )
    return (
        f"{_number_word(checked)} of the {_number_word(total).lower()} were retrieved "
        f"and checked against the document they name. For the remaining "
        f"{_plural(total - checked, 'source')}, {outstanding}; which is which is "
        "recorded against each entry in the evidence appendix."
    )


def _section_eight(record: ResearchRecord, briefs: Sequence[IdeaBrief]) -> _Draft:
    manifest = record.manifest
    fatal = [
        (brief.title, review)
        for brief in briefs
        for review in brief.reviews
        if review.score <= 2
    ]
    core = [
        # This opened by promising "what this run established", which section 1 has
        # already denied outright and the paragraph immediately below denies again.
        # It was also true of every run word for word, so it told a reader nothing
        # about the one in front of them.
        "This run established nothing about the world. What follows is the state of "
        "the idea space it produced: which proposal leads, which reviews went "
        "against the field, and where the meta-review's account of the round and "
        "the reviews themselves disagree."
    ]
    if record.citations:
        core.append(
            # The count used to be stated here as the number of sources cited, taken
            # from the citation registry. A source is numbered the moment the builder
            # reaches for it and the paragraph holding it can still be cut afterwards,
            # so the figure ran ahead of the reference list more often than not. What
            # survives into References is settled after this sentence is written.
            "The sources this report draws on are listed under References. "
            + _reference_standing(record)
        )
    else:
        core.append(
            "Discovery resolved no external source for this goal, which is the single "
            "most consequential finding in this report. Every idea below is an "
            "inference from the framing of the question and from general "
            "methodological practice. None of them is evidence-backed, and the "
            "ranking that separates them is therefore a ranking of internal "
            "coherence, not of empirical support."
        )
    if briefs:
        leader = briefs[0]
        core.append(
            # The claim and the predictions used to be reprinted here word for word
            # from the leading idea's own section, a hundred-odd lines above. This
            # section is meant to say what the run concluded, not to restate the
            # proposal: what belongs here is the outcome that made it the leader,
            # and a pointer to where the proposal itself is set out in full.
            f"{leader.title} is the strongest proposal this run produced: it "
            f"finished rank {leader.rank} on an Elo of {leader.elo}"
            + (
                f" with {_plural(leader.wins, 'win')} from "
                f"{_plural(len(leader.matches), 'match')}"
                if leader.matches
                else ""
            )
            + ", and its claim, predictions and falsifier are set out under its own "
            f"heading in {DEEP_DIVE_CHAPTER}, below."
            # The sentence promised the predictions were above and then printed all
            # of them here anyway, which made this the third place in the report to
            # carry the same joined list. The risks stay, because a findings section
            # that names a recommendation has to name its cost -- but this is not the
            # only place they appear: the section-4 grid carries the first of them and
            # the deep dive carries them under its issues heading. Claiming otherwise
            # here, as this used to, was simply false.
            + (
                # "Its exposure is that ..." forces every risk into a that-clause, and
                # a specialist writing risks does not write clauses -- half of them are
                # bare noun phrases, so the sentence came out as "is that ... and anode
                # poisoning by dissolved Zn2+ ions". A colon takes either.
                " What it would risk, in full: "
                f"{_spliced(leader.facts['Principal risks'])}."
                if _stated(leader.facts, "Principal risks")
                else " No material risk was recorded against it, which is a silence "
                "in the proposal rather than a finding that it carries none."
            )
        )
    if fatal:
        # The closing sentence used to be a standing rule -- "where the same idea
        # attracts several such reviews, the shortfall is in the evidence base rather
        # than in the idea's construction" -- introduced by a condition the report
        # never checked. On a live run no idea had drawn more than one such review, so
        # the rule fired against nothing, and the diagnosis it offered was one the
        # reviewers had not made: they had faulted the construction. Which ideas drew
        # more than one is a count, so it is counted and stated.
        repeated = Counter(title for title, _ in fatal)
        several = sorted(title for title, count in repeated.items() if count > 1)
        # Six of the seven titles just listed, listed again in the next sentence, is a
        # wall of the same names twice over: the sentence says which of the affected
        # ideas drew more than one such review, so where that is nearly all of them it
        # is shorter and clearer to name the ones it is not.
        once = sorted(title for title, count in repeated.items() if count == 1)
        drew = (
            # "Every one of them" needs a them to be every one of: where a single idea
            # is affected at all, the sentence names it.
            "Every one of them drew"
            if not once and len(several) > 1
            else "All of them except " + _joined_titles(once) + " drew"
            if once and len(once) < len(several)
            else _joined_titles(several)
            + (" drew" if len(several) == 1 else " each drew")
        )
        core.append(
            # The band used to be described as covering a rejection and evidence
            # too thin to judge on alike. Under the scale section 2 sets out, a
            # confident finding that the evidence is too thin scores three and is
            # not in this band at all, while a review of any kind that records a
            # fatal flaw is capped into it.
            f"{_opening(len(fatal), 'review')} closed at two or below, the band that "
            "covers a rejection, a hesitant finding that the evidence is too thin "
            "to judge on, and any review at all that recorded a fatal flaw. "
            # Where every idea is affected, naming them is seven long titles to say
            # what one clause says, and the next sentence then names most of them
            # again. The list earns its length only where it excludes something.
            + (
                "No idea in the run escaped the band."
                if len(repeated) == len(briefs) > 1
                else "The affected ideas are "
                + _sentence_of_titles(sorted(repeated), fallback="none.")
            )
            + (
                f" {drew} more than one such review, which is a judgement several "
                "reviewers reached separately rather than one reviewer's reservation."
                if several
                else " No idea drew more than one of them, so each of these is a "
                "single reviewer's judgement and the reviews that sit beside it "
                "should be read before it is treated as settled."
            )
        )
    if manifest and manifest.unresolved_fatal_flaw_candidate_ids:
        excluded = list(
            dict.fromkeys(
                record.ranked_id(item)
                for item in manifest.unresolved_fatal_flaw_candidate_ids
            )
        )
        recorded = record.recorded_fatal_flaw_ids
        # An idea a person withdrew is not the meta-review's to exclude, and the
        # meta-review lists it anyway: on the governance run it named a hypothesis a
        # named adjudicator had pulled from the population three stages earlier. Filed
        # under the model's exclusions it read as one more automatic decision, and the
        # one decision in the run that a human actually took disappeared into a list.
        pulled = {item.candidate_id for item in record.withdrawals}
        withdrawn = [item for item in excluded if item in pulled]
        excluded = [item for item in excluded if item not in pulled]
        # The exclusion list is the meta-review's own account of the review round. It
        # is checked against the reviews rather than repeated, because a fatal flaw
        # asserted against an idea no reviewer faulted is a claim about that idea that
        # the report would be inventing, and an omitted one is a warning withheld.
        unfounded = [item for item in excluded if item not in recorded]
        # A withdrawn idea is not in contention, so a flaw against it is not open
        # against anything. Left in, it printed "the flaw stands open against an idea
        # still in contention" about a hypothesis a person had already pulled.
        missing = sorted(recorded - set(excluded) - pulled)
        if excluded:
            core.append(
                # "because a fatal flaw remains unresolved" asserted the flaw in the
                # report's own voice, and the check printed directly below it withdrew
                # the assertion for one of the four ideas it had just been made about.
                # The exclusion is the meta-review's; whether the record bears out its
                # stated reason is what the next two paragraphs are for, so this one
                # says who excluded what and on what stated grounds.
                # Where the exclusion takes the whole field, the list excludes nothing
                # and seven long titles say what one clause says.
                (
                    "The meta-review excluded every ranked idea from any "
                    "recommendation, stating that an unresolved fatal flaw stands "
                    "against each."
                    if len(excluded) == len(briefs) > 1
                    else "The meta-review excluded the following from any "
                    "recommendation, stating that an unresolved fatal flaw stands "
                    "against each: "
                    + _sentence_of_titles(
                        [record.title_for(item) for item in excluded], fallback="none."
                    )
                )
                + " Exclusion here is procedural and not reversible by a better score "
                "elsewhere. "
                # The two paragraphs below are emitted only on a mismatch, so on a run
                # where the reviews bear the meta-review out this promised a check and
                # then went on to the withdrawal. A check that found nothing has a
                # result, and it is the one the reader wants: the exclusions hold.
                + (
                    "Whether the reviews carry the flaw it names is checked below."
                    if unfounded or missing
                    else "The reviews carry the flaw in every case: one was recorded "
                    "against each idea named here"
                    + (
                        "."
                        if len(excluded) == len(briefs) > 1
                        else ", and against no idea the meta-review left standing."
                    )
                )
            )
        if withdrawn:
            core.append(
                "The meta-review also listed "
                + _sentence_of_titles(
                    [record.title_for(item) for item in withdrawn], fallback="none."
                ).rstrip(".")
                + (" as excluded, which " if len(withdrawn) == 1 else ", which ")
                + ("it was" if len(withdrawn) == 1 else "they were")
                + ", but not by the meta-review: "
                + ("it was" if len(withdrawn) == 1 else "they were")
                + " withdrawn from the population by a named person, and the "
                "adjudication is set out with the flaw and the reason given under "
                "Governance adjudications below. That is a human decision on the "
                "record, not a summary-stage one."
            )
        if unfounded:
            core.append(
                "No reviewer recorded a fatal flaw against "
                + _sentence_of_titles(
                    [record.title_for(item) for item in unfounded], fallback="none."
                )
                + " The exclusion is the meta-review's own judgement and the reviews "
                "below do not carry it, so it should be read as a decision taken at "
                "the summary stage rather than as a finding."
            )
        if missing:
            core.append(
                "A fatal flaw was recorded against "
                + _sentence_of_titles(
                    [record.title_for(item) for item in missing], fallback="none."
                )
                + " The meta-review did not exclude "
                + ("it" if len(missing) == 1 else "them")
                + " on that basis, so the flaw stands open against an idea still in "
                "contention. " + _open_flaw_consequence(record, missing)
            )
    extra = []
    if record.evolution and record.evolution.records:
        extra.append(
            f"Evolution refined {_plural(len(record.evolution.records), 'idea')} over "
            f"{_plural(record.evolution.records[-1].round_number, 'round')} and stopped "
            f"because {_spliced(_sentence(record.evolution.stop_reason or 'the budget was reached'))}. "
            + (
                # The line here used to be that a refined idea inherits its parent's
                # reviews and so carries no confirmation of its own. That is the right
                # caution when nothing re-checked the rewrite, and simply false when
                # something did: the cycle carries a re-review per revision.
                f"The rewrites were checked again rather than left on the parent's "
                f"reviews: {_plural(len(record.evolution.rereviews), 're-review')} "
                f"{'was' if len(record.evolution.rereviews) == 1 else 'were'} recorded "
                "against them, each set out with its idea under Revised Form "
                "Recommended. Those reviews are the same specialists reading their own "
                "answered objections, so they confirm the objections were addressed "
                "rather than that the idea now holds."
                if record.evolution.rereviews
                else "No re-review of the rewrites is on the record, so a refined idea "
                "carries its parent's review history and nothing of its own."
            )
        )
    if record.evidence and record.evidence.limitations:
        extra.append(
            "Verification recorded its own limits explicitly. "
            + _join(record.evidence.limitations, fallback="None recorded.")
            + " These are the sentences a reader should quote back when asked whether "
            "the report supports a decision."
        )
    return _Draft(8, "Key Findings and Unexpected Connections", core, extra)


def _recommended_overlaps(
    record: ResearchRecord, recommended_ids: Sequence[str]
) -> list[tuple[str, list[str]]]:
    """Each cluster the recommendation drew more than one idea from, and which ideas.

    Main Research Directions tells the reader that two ideas in one cluster fail for
    the same reason, so funding both buys less than the pair of scores suggests. On a
    live run the meta-review then recommended two ideas out of one cluster, and the
    report had stated both things and reconciled neither -- leaving a reader to
    either notice the overlap themselves or take four recommendations for four bets.

    Naming it is the reconciliation. The recommendation is the meta-review's and is
    not second-guessed here; what changes is that the report says what it costs.
    """
    chosen = list(dict.fromkeys(recommended_ids))
    overlaps = []
    for cluster in record.landscape.clusters if record.landscape else []:
        members = {record.ranked_id(item) for item in cluster.candidate_ids}
        # Ordered by the recommendation rather than by the cluster, so the titles run
        # in the order the reader has already met them in.
        shared = [item for item in chosen if item in members]
        if len(shared) > 1:
            overlaps.append((cluster.name, [record.title_for(item) for item in shared]))
    return overlaps


def _recommendation_grounding(
    record: ResearchRecord, recommended_ids: Sequence[str]
) -> str:
    """What the recommended ideas rest on, said where the recommendation is made.

    The Evidence integrity appendix carries this per idea, and on a live run it read
    that the leading recommended idea "cites no evidence at all ... Its claim is a
    conjecture" while the other three rested on evidence discovered and never checked.
    This section's only qualifications were protocol drafting and outside review, so a
    reader who acted on the recommendation without reaching the appendix would not know
    the top of it was an unevidenced conjecture. One sentence, and the cases stay in the
    appendix: the point here is the standing of the set, not a second copy of the list.
    """
    ratings = record.tournament.ratings if record.tournament else {}
    ordered = sorted(
        dict.fromkeys(recommended_ids),
        key=lambda item: (-ratings.get(record.ranked_id(item), 0.0), item),
    )
    supports = {
        item: (
            record.evidence_support[item].support
            if item in record.evidence_support
            else "unknown"
        )
        for item in ordered
    }
    if not ordered or all(support == "unknown" for support in supports.values()):
        return ""
    verified = [
        item for item in ordered if supports[item] in {"grounded", "partially_grounded"}
    ]
    if len(verified) == len(ordered):
        return (
            " Each of these rests on evidence that was retrieved and checked against "
            "its source, which is the strongest grounding this workflow records."
        )
    total = _number_word(len(ordered)).lower()
    if verified:
        lead = (
            f" {_number_word(len(verified))} of the {total} "
            + ("rests" if len(verified) == 1 else "rest")
            + " on verified evidence and the "
            + ("other does" if len(ordered) - len(verified) == 1 else "others do")
            + " not."
        )
    else:
        lead = f" None of the {total} rests on verified evidence."
    if supports[ordered[0]] == "uncited":
        lead += (
            f" {record.title_for(ordered[0])}, which leads the recommendation, cites "
            "none at all: it is a conjecture, and carrying it is a bet on an argument "
            "rather than on a finding."
        )
    return (
        lead + " Evidence integrity in the appendix names which case applies to each."
    )


def _rereview_sentence(record: ResearchRecord, recommended_ids: Sequence[str]) -> str:
    """What the run did with the rewrites after it made them.

    This sentence used to say the revisions "were not re-reviewed or re-ranked", which
    was untrue in both halves on every run that reaches it: the evolution cycle carries
    a re-review per revision and a further ranking round, and the report was telling a
    reader to treat checked work as unchecked. Which way the error runs matters -- it
    understates the run, so a reader discounts a recommendation the record supports --
    but either way the sentence has to be read off the record rather than assumed.
    """
    reviews = [
        review for item in recommended_ids for review in record.rereviews_of(item)
    ]
    rounds = len(record.evolution.ranking_history if record.evolution else [])
    if not reviews and not rounds:
        return (
            "Neither check was run again on the rewrites: no re-review and no further "
            "ranking round is on the record, so what changed has to be read before the "
            "recommendation is acted on."
        )
    parts = []
    if reviews:
        verdicts = Counter(review.recommendation for review in reviews)
        leading, count = verdicts.most_common(1)[0]
        verdict = _RECOMMENDATION_SHORT.get(leading, leading)
        agreement = (
            f"every one of which returned {verdict}"
            if count == len(reviews)
            else f"{_number_word(count).lower()} of which returned {verdict}"
        )
        # The clause does not pick out which re-reviews are meant -- there is only the
        # one set -- so it needs the comma. Without it, "eight re-reviews every one of
        # which returned advance" reads for a beat as though some other re-reviews
        # returned something else.
        parts.append(f"{_plural(len(reviews), 're-review')}, {agreement}")
    if rounds:
        parts.append(f"{_plural(rounds, 'further ranking round')}")
    faulted = [review for review in reviews if review.fatal_flaws]
    tail = (
        f" {_number_word(len(faulted))} of those re-reviews recorded a fatal flaw "
        "against the rewrite, which is set out with the idea."
        if faulted
        else ""
    )
    return f"Both checks were run again on the rewrites: {_series(parts)}.{tail}"


def _post_evolution_reordering(
    record: ResearchRecord,
    recommended_ids: Sequence[str],
    briefs: Sequence[IdeaBrief],
) -> str:
    """Say so when the post-evolution round disagrees with the order section 4 prints.

    A reader carries the section-4 order forward; if the round that ranked the text
    actually being recommended put it in a different order, the ranking under the
    recommendation is not the ranking they are holding.
    """
    settled = [
        item for item in record.post_evolution_order if item in set(recommended_ids)
    ]
    # Through the lineage on both sides. The briefs are built from the live population,
    # which after evolution holds the rewrites, so their ids are the revision ids and
    # never appear in either list above -- both of which the caller has already
    # resolved back to the ranked field. On a live run that made the second list empty
    # and the paragraph read "ranked on the proposals ... they come out ." while the
    # first list named four ideas.
    ranked = list(
        dict.fromkeys(
            resolved
            for brief in briefs
            if (resolved := record.ranked_id(brief.candidate_id)) in set(settled)
        )
    )
    # Two orders are compared, so there is nothing to say unless both are orders.
    if len(settled) < 2 or len(ranked) < 2 or settled == ranked:
        return ""
    return (
        "That round did not agree with section four about the order. Ranked on the "
        "revised text they come out "
        + _joined_titles([record.title_for(item) for item in settled])
        # A full stop between the two orders, not a semicolon. _joined_titles separates
        # multi-word titles with semicolons, so the mark that ended the first list and
        # the mark inside it were the same one, and the second clause read as a fifth
        # item: "... Thicker than 15 nm; ranked on the proposals ...", eighty-six words
        # from the full stop that opened it.
        + ". Ranked on the proposals, as the standings in section five set them out, "
        "they come out "
        + _joined_titles([record.title_for(item) for item in ranked])
        + ". Those standings rank what was proposed; this order ranks what is being "
        "recommended, and where the two differ it is this one that applies."
    )


def _leader_blockers(record: ResearchRecord, leader: IdeaBrief) -> list[str]:
    """What stands against the top idea, which is what has to be closed before it runs.

    Read twice: once by the paragraph that sends a reader to the least weak idea, and
    once by the paragraph that names the first thing to do, so that the second does not
    call its step first while the first says something else comes before all of it.
    """
    manifest = record.manifest
    excluded = {
        record.ranked_id(item)
        for item in (manifest.unresolved_fatal_flaw_candidate_ids if manifest else [])
    }
    against = []
    if leader.candidate_id in record.recorded_fatal_flaw_ids:
        against.append("a reviewer recorded a fatal flaw against it")
    if leader.candidate_id in excluded:
        against.append("the meta-review excluded it from any recommendation")
    return against


def _least_weak(record: ResearchRecord, leader: IdeaBrief) -> str:
    """Where to spend the next increment when nothing was recommended.

    The idea at the top of a ranking nobody would act on is still the idea at the top,
    so it is still the answer to where to look next -- but it may be there carrying a
    fatal flaw or a meta-review exclusion, and a paragraph that sends the reader to it
    without saying so has hidden the reason there is no recommendation in the first
    place.
    """
    against = _leader_blockers(record, leader)
    standing = (
        f" That is the standing of the whole set rather than a clearance for this one: "
        f"{_series(against)}, and closing that comes before any of the work below."
        if against
        else ""
    )
    return (
        f"No idea cleared the bar for an unconditional recommendation. {leader.title} "
        "is the least weak of the set and is the sensible place to spend the next "
        "increment of effort, but it should be funded as an evidence-gathering "
        f"exercise rather than as a test of the idea itself.{standing}"
    )


def _section_nine(record: ResearchRecord, briefs: Sequence[IdeaBrief]) -> _Draft:
    manifest = record.manifest
    # Resolved back to the ranked field. The meta-review names whichever revision was
    # current when it ran, and those ids carry claims the reader has not been shown.
    resolved_ids = list(
        dict.fromkeys(
            record.ranked_id(item)
            for item in (manifest.recommendation_candidate_ids if manifest else [])
        )
    )
    # Anything still holding a revision's own id did not resolve. Its title is derived
    # from the rewritten claim, so on a live run this section recommended four ideas
    # under four names that appear nowhere else in the report -- one of them stating
    # the opposite of the ranked idea it was a rewrite of, because the rewrite had
    # reversed the claim and the title follows the claim. Reported as unmatched now,
    # rather than presented as a fifth, sixth, seventh and eighth idea.
    unresolved = {
        item.candidate.id
        for item in (record.evolution.records if record.evolution else [])
    }
    recommended_ids = [item for item in resolved_ids if item not in unresolved]
    unmatched_ids = [item for item in resolved_ids if item in unresolved]
    recommended = [record.title_for(item) for item in recommended_ids]
    # Only where the idea's own section will carry the rewrite. This paragraph sends
    # the reader to a Revised Form heading under each recommended idea, and the block
    # that prints it stands down when the rewrite changed none of the fields the
    # report shows -- so the promise was made and nothing kept it.
    printed = {brief.candidate_id for brief in briefs if brief.revised_form}
    revised = [
        (item, record.title_for(item), revision)
        for item in recommended_ids
        if item in printed and (revision := record.revision_of(item)) is not None
    ]
    core = [
        "The recommendation below is a proposal for what to do next, conditional on "
        "the verification steps named with it. It is not an approval, and no part of "
        "this workflow can supply one."
    ]
    if recommended:
        core.append(
            "The meta-review recommends carrying "
            + _sentence_of_titles(recommended, fallback="no idea.")
            + " Each should be taken to protocol drafting rather than to execution, "
            "and the protocol should be reviewed by a domain specialist who was not "
            "involved in producing this report."
            + _recommendation_grounding(record, recommended_ids)
        )
        # Section 8 checks the meta-review's exclusions against the reviews and, on a
        # live run, found a fatal flaw recorded against an idea the meta-review had
        # not excluded -- and then recommended. The report told the reader in one
        # section that no work may proceed and in the next that the idea should go to
        # protocol drafting, and reconciled neither. The recommendation is the
        # meta-review's and is not overridden here; what changes is that the flaw
        # travels with it instead of being left a section behind.
        flagged_ids = [
            item for item in recommended_ids if item in record.recorded_fatal_flaw_ids
        ]
        flagged = sorted(record.title_for(item) for item in flagged_ids)
        if flagged:
            core.append(
                _sentence_of_titles(flagged, fallback="none.").rstrip(".")
                + (" is" if len(flagged) == 1 else " are")
                + " recommended while carrying the fatal flaw a reviewer recorded "
                + ("against it" if len(flagged) == 1 else "against each of them")
                # This pointed at section eight, which reports that the flaw stands
                # open and which review recorded it but never prints what the flaw
                # says -- and prints nothing at all unless the meta-review named an
                # exclusion list to be checked, so on a run with no exclusions the
                # sentence sent the reader to a paragraph that was not there. The
                # flaw text is under the idea's own Deep Verification on every run,
                # because every ranked idea gets a section and that block prints the
                # flaws before the objections.
                + ", printed in full under Deep Verification in "
                + ("its own section" if len(flagged) == 1 else "each of their sections")
                + " below. The meta-review did not address "
                + ("that flaw" if len(flagged) == 1 else "those flaws")
                + ", so closing "
                + ("it" if len(flagged) == 1 else "them")
                # What closing it takes depends on what kind of flaw it is -- a safety
                # finding and a novelty finding are both fatal to a review's score and
                # are not both a reason to stop -- and that is said with the flaw in
                # the idea's own section, which this sentence now points the reader at.
                # Repeating it here printed the same clause twice on one page.
                + " comes before the protocol is drafted, not after."
            )
        overlaps = _recommended_overlaps(record, recommended_ids)
        if overlaps:
            core.append(
                "Part of what is recommended is one bet rather than several. "
                + " ".join(
                    f"{_joined_titles(titles)} sit in the {name} cluster and rest on "
                    "the single mechanism named for it under Main Research Directions "
                    "above, so they stand or fall together."
                    for name, titles in overlaps
                )
                + " Carrying them together is still defensible, since the protocols "
                "differ and the cheapest of them may settle the mechanism for the "
                "rest. What it is not is diversification: the shared mechanism is the "
                "thing to test first, and a result against it takes more than one idea "
                "off this list at once."
            )
    if unmatched_ids:
        core.append(
            "The meta-review also recommended "
            + _plural(len(unmatched_ids), "candidate")
            + " this report cannot match to any idea it ranked: "
            + _join([f"`{item}`" for item in unmatched_ids], fallback="none").rstrip(
                "."
            )
            + ". The identifier belongs to a rewrite whose parent the evolution stage "
            "did not record, so nothing here says which of the ranked ideas "
            + ("it is" if len(unmatched_ids) == 1 else "they are")
            + " a rewrite of. "
            + (
                "That recommendation"
                if len(unmatched_ids) == 1
                else "Those recommendations"
            )
            + " should be read off the evolution stage's own output rather than "
            "from this report."
        )
    if revised:
        core.append(
            "The recommendation is for the revised form of each of these, not the form "
            "ranked above. Evolution rewrote them after the tournament, and the revised "
            "text is set out under Revised Form Recommended in each idea's own section: "
            "that is what would be carried, and it is not the text ranked in section "
            "four. "
            + _rereview_sentence(record, recommended_ids)
            + " "
            # Every round of the lineage, not the last one. The change log is the only
            # place a reader is told what the rewrite did, and printing the final
            # round's changes alone described a two-round rewrite by its smaller half:
            # "revised to version 3: specified H14-grade HEPA filtration" over an idea
            # whose first round had changed the coating material and the loading.
            + " ".join(
                f"{title} was revised to version {revision.candidate.version}: "
                + _spliced(
                    _join(
                        list(
                            dict.fromkeys(
                                change
                                for item in record.revisions_of(candidate_id)
                                for change in item.changes
                            )
                        ),
                        fallback="no change was recorded",
                    )
                )
                + "."
                for candidate_id, title, revision in revised
            )
        )
        reordering = _post_evolution_reordering(record, recommended_ids, briefs)
        if reordering:
            core.append(reordering)
    # Keyed on there being no recommendation, which is what the paragraph says. It used
    # to be the else of "if revised", so a run that recommended four ideas and revised
    # none of them printed "no idea cleared the bar for an unconditional recommendation"
    # directly under the sentence recommending them, and then named a fifth idea -- the
    # top of the ranking, which on that run the meta-review had excluded for an
    # unresolved fatal flaw -- as the place to spend the next increment of effort.
    blocked = bool(not recommended and briefs and _leader_blockers(record, briefs[0]))
    obligations = (
        _join(record.plan.governance_requirements, fallback="")
        if record.plan and record.plan.governance_requirements
        else ""
    )
    if not recommended and briefs:
        core.append(_least_weak(record, briefs[0]))
    if manifest and manifest.evidence_that_would_change_decision:
        core.append(
            "The decision is sensitive to a short list of specific evidence. "
            + _join(
                manifest.evidence_that_would_change_decision,
                fallback="No decision-changing evidence was identified.",
            )
            + " Obtaining any one of these is worth more than another round of "
            "generation, because it would move the ranking rather than lengthen it."
        )
    if briefs:
        # The go/no-go the reader is sent to run has to belong to the form that would
        # be carried. Quoted from the ranked candidate, it gave the leading idea's
        # pre-revision test -- "within 5% via TGA or ICP-OES" -- two paragraphs after
        # the report said the revised form is what is recommended, and the rewrite had
        # dropped TGA.
        leading = record.revision_of(briefs[0].candidate_id)
        ranked_tests = briefs[0].facts["Go/no-go tests"]
        tests = (
            _idea_facts(leading.candidate)["Go/no-go tests"]
            if leading is not None
            else ranked_tests
        )
        core.append(
            # "its own go/no-go test" and "Running that test first" asserted a count
            # the field cannot supply: it is a list, and on a live run it held two.
            "The immediate next step for the leading idea is the go/no-go work its "
            "specialist set down in advance"
            + (
                ", as the rewrite recommended below states it rather than as the "
                "ranked form did. "
                if tests != ranked_tests
                else ". "
            )
            + tests
            # "Running that first" against a paragraph two above it saying that
            # closing the flaw and the exclusion "comes before any of the work below",
            # and a paragraph below it saying the governance obligations have to be
            # discharged before any physical step: three sentences, each naming a
            # different thing to do first. This is the step that is first among the
            # work; the others are what has to be cleared before the work starts, so
            # it is the one that gives way.
            + (
                " Running that first is what converts this report from a set of "
                "proposals into a decision."
                if not (blocked or obligations)
                else " Running that, once "
                + _names(
                    (["the flaw and the exclusion above are closed"] if blocked else [])
                    + (
                        ["the governance obligations below are discharged"]
                        if obligations
                        else []
                    )
                )
                + ", is what converts this report from a set of proposals into a "
                "decision."
            )
        )
    if obligations:
        # Core, not elaboration. This is the only place the report states what has to
        # be cleared before any of the work below may be started, and as an optional
        # paragraph it was dropped from every live report the budget filled.
        #
        # The obligations are noun phrases -- "adherence to laboratory safety
        # protocols", "proper disposal procedures" -- so set after a full stop they
        # made a sentence with no verb in it. The colon hands them the lead-in's.
        core.append(
            "Before any physical, clinical or data-access step, the governance "
            "obligations attached to the goal have to be discharged by a named owner: "
            f"{_spliced(obligations)}. None of these can be discharged by the workflow "
            "itself, and an automatic approval recorded in it satisfies none of them."
        )
    extra = []
    for brief in briefs[1:5]:
        extra.append(
            f"If the leading idea fails its first test, {brief.title} is the next "
            f"place to look. {brief.facts['Core idea']} What the specialist set down "
            f"to decide it: {_spliced(brief.facts['Go/no-go tests'])}."
        )
    extra.append(
        "Finally, the ideas that were not recommended should be retained rather than "
        "discarded. This run ranked them on an evidence base that it has itself "
        "described as thin; a single retrieved source can reorder the table, and the "
        "cost of re-generating a discarded idea is higher than the cost of keeping it."
    )
    return _Draft(9, "Recommendations and Next Steps", core, extra)


# The reference reports hold their narrative body to a consistent length. Sections are
# written as a mandatory core plus optional elaboration so the total can be fitted
# without cutting a sentence in half or padding with filler.
NARRATIVE_WORD_FLOOR = 4300
NARRATIVE_WORD_CEILING = 4600


def _fit_word_budget(
    drafts: Sequence[_Draft],
    *,
    floor: int = NARRATIVE_WORD_FLOOR,
    ceiling: int = NARRATIVE_WORD_CEILING,
) -> list[NarrativeSection]:
    """Seat every core paragraph, then add elaboration round-robin up to the budget.

    Elaboration is fitted towards the middle of the band rather than the floor: a
    report that lands one paragraph above the minimum is one edit away from breaching
    it, and the floor exists to stop the narrative thinning out.
    """
    target = (floor + ceiling) // 2
    sections = [
        NarrativeSection(
            number=draft.number,
            title=draft.title,
            paragraphs=list(draft.core),
            subsections=list(draft.subsections),
            grids=list(draft.grids),
        )
        for draft in drafts
    ]
    total = sum(
        len(paragraph.split())
        for section in sections
        for paragraph in _section_prose(section)
    )
    queues = [list(draft.extra) for draft in drafts]
    while total < target and any(queues):
        seated = False
        for section, queue in zip(sections, queues, strict=True):
            if not queue or total >= target:
                continue
            paragraph = queue.pop(0)
            length = len(paragraph.split())
            if total + length > ceiling:
                continue
            section.paragraphs.append(paragraph)
            total += length
            seated = True
        if not seated:
            break
    return sections


def _review_summary(briefs: Sequence[IdeaBrief]) -> list[str]:
    scores: dict[str, list[int]] = {}
    for brief in briefs:
        for review in brief.reviews:
            scores.setdefault(review.section, []).append(review.score)
    summary = [
        f"{criterion}: mean {sum(values) / len(values):.1f} of five across "
        f"{_plural(len(values), 'review')}, range {min(values)} to {max(values)}."
        for criterion, values in scores.items()
    ]
    if not summary:
        summary.append("No review was recorded against any idea in this run.")
    return summary


def _without_restatements(questions: Sequence[str]) -> list[str]:
    """The questions, with later rewordings of an earlier one dropped.

    Four specialists write into this list from their own payloads, and they reach
    the same gap independently. A live run printed "Empirical data demonstrating
    the electrochemical stability of self-healing polymers at voltages >4.0V" and
    then, three bullets later, "Missing empirical data on the electrochemical
    stability of self-healing polymers at high cathode voltages (>4.0V)". Exact-
    string dedup cannot see that; overlap of content words can, and the first
    phrasing is kept because it is the one the reader has already read.
    """
    kept: list[str] = []
    seen: list[set[str]] = []
    for question in questions:
        words = _content_words(question)
        if not words:
            continue
        if any(len(words & earlier) / len(words) >= 0.7 for earlier in seen):
            continue
        kept.append(question)
        seen.append(words)
    return kept


def _constraint_coverage(record: ResearchRecord, constraints: Sequence[str]) -> str:
    """Which of the goal's constraints a reviewer actually wrote about, and which none did.

    This used to count the safety and governance reviews and then assert that "none
    of those reviewers was asked to check the constraints above and none records
    having done so", which was two claims and both were wrong. It was the wrong set
    of reviews -- on the live run it was the feasibility reviewers who wrote about
    the uncoated control cells and the sample size, criterion by criterion, in all
    eight of their reviews -- and it was the wrong verdict, because section 6 of the
    same report goes on to name a constraint a reviewer found violated. A report
    that tells the reader nothing was checked, and then reports a failed check, has
    spent the credibility of both statements.

    No stage screens a candidate against a constraint, so this cannot claim a
    verdict. What it can do is say which constraints a reviewer's own words reach:
    the overlap is on content words, the same test the open-questions list uses to
    spot a reworded duplicate, and it is reported as coverage rather than as
    compliance.
    """
    reviews = record.shown_reviews
    if not reviews:
        return (
            "No review was run either, so nothing in this report speaks to whether "
            "any idea complies."
        )
    held = f"What the record does hold is {_plural(len(reviews), 'review')} of the ideas, printed in full under each."
    # The judges argued the ideas against the goal, and what they wrote about the
    # requirements was read by nothing in this section. A live report told the reader
    # "nothing any reviewer wrote reaches constraint three, so on that one the report
    # is silent" -- above a tournament transcript in which a judge had written that
    # both hypotheses fail the constraint requiring exact charge and discharge rates,
    # voltage windows and temperature, quoting it. Silence was the one thing the
    # record did not hold.
    turns = [
        text
        for comparison in (record.tournament.comparisons if record.tournament else [])
        for text in (*comparison.debate_turns, comparison.rationale)
        if text.strip()
    ]
    spoken: dict[int, set[str]] = {}
    counts: Counter[int] = Counter()
    argued: Counter[int] = Counter()
    # Which turns reached a constraint, not how many constraint-hits there were. A turn
    # that argues rate, voltage window and temperature reaches three constraints, and
    # summing the per-constraint tallies counted it three times: a run of 27 turns was
    # reported as 33, more turns than the transcript below it holds.
    reaching: set[int] = set()
    for index, constraint in enumerate(constraints, start=1):
        wanted = _content_words(constraint) - _CONSTRAINT_DIRECTIVES
        if not wanted:
            continue
        for review in reviews:
            said = " ".join([*review.findings, *review.objections, *review.fatal_flaws])
            if _reaches(wanted, said):
                spoken.setdefault(index, set()).add(
                    CRITERION_SECTIONS.get(review.criterion, review.criterion)
                )
                counts[index] += 1
        hit = {
            position for position, turn in enumerate(turns) if _reaches(wanted, turn)
        }
        argued[index] += len(hit)
        reaching |= hit
    reached_in_debate = sorted(index for index, count in argued.items() if count)
    debated = (
        " The tournament debates reach "
        + _numbered_constraints(reached_in_debate)
        + f", in {_plural(len(reaching), 'turn')} across the matches. A judge "
        "argues two ideas against the goal rather than screening one against a "
        "list, so what a turn says about a requirement is said about the pair in "
        "front of it; the turns are quoted in full under each idea's Tournament "
        "section."
        if reached_in_debate
        else ""
    )
    if not spoken:
        return (
            f"{held} Not one of them writes about any of the constraints above.{debated}"
            + (
                " Nothing in this report speaks to whether an idea complies."
                if not reached_in_debate
                else ""
            )
        )
    reached = _series(
        [
            _numbered_constraints([index])
            + f", in {_plural(counts[index], 'review')} under "
            + (
                # Not _series: it folds every item after the first to lower case,
                # which turned a pair of section headings into "the Correctness, and
                # novelty reviews" -- a heading the reader cannot then find. Past two
                # headings the list stops being a pointer and becomes a recital, so
                # it is counted instead.
                _joined_titles(sorted(spoken[index]))
                if len(spoken[index]) <= 2
                else _number_word(len(spoken[index])).lower()
                + " of the "
                + _number_word(len(REVIEW_SECTIONS)).lower()
                + " review headings"
            )
            for index in sorted(spoken)
        ]
    )
    untouched = [
        index for index in range(1, len(constraints) + 1) if index not in spoken
    ]
    silent = [index for index in untouched if index not in reached_in_debate]
    if not untouched:
        tail = " Every constraint is reached by at least one review."
    elif silent == untouched:
        # The debates added nothing to what the reviews missed, so the gap and the
        # silence are one fact. Stated as two sentences they read as two findings:
        # "nothing any reviewer wrote reaches constraints two and three. Constraints
        # two and three are reached by neither."
        tail = (
            f" Nothing any reviewer wrote reaches {_numbered_constraints(untouched)}, "
            f"so on {'that one' if len(untouched) == 1 else 'those'} this report shows "
            "no coverage."
        )
    else:
        tail = (
            f" Nothing any reviewer wrote reaches {_numbered_constraints(untouched)}."
        )
    tail += debated
    if silent and silent != untouched:
        tail += (
            f" {_numbered_constraints(silent).capitalize()} "
            + ("is" if len(silent) == 1 else "are")
            + " reached by neither, so there this report shows no coverage."
        )
    if untouched:
        # The unreached constraints are the output of a content-word match, and the
        # paragraph asserted their absence as a property of the record: "so on that one
        # the report is silent". Silence is a strong claim to rest on a wording test,
        # and it is the half of this paragraph a reader is most likely to act on.
        tail += (
            " The match is on wording, so a reviewer who addressed a constraint in "
            "other terms than the goal states it in would not be counted here."
        )
    return (
        f"{held} Read against the constraints, "
        + _numbered_constraints(sorted(spoken))
        + " "
        + ("is" if len(spoken) == 1 else "are")
        + f" reached by a reviewer's own words: {reached}. None of those reviewers was "
        "asked to check a constraint and none records a verdict on one, so this says "
        f"where to look rather than whether an idea complies.{tail}"
    )


# The imperative scaffolding every constraint is written with. "Must", "specify" and
# "statistically justified" are how a plan phrases a requirement, not what the
# requirement is about, and counting them as content words held the overlap test
# below its threshold on constraints eight reviewers had plainly written about.
_CONSTRAINT_DIRECTIVES = frozenset(
    """must shall should require required include included define defined defining
    specify specified specifying specific exact exactly statistically justified
    provide provided ensure per each""".split()
)


def _reaches(wanted: set[str], text: str) -> bool:
    """Whether a passage is written about the constraint these words came from.

    Two of the constraint's own five-odd words, which is what "uncoated control" is
    out of "uncoated control cells for direct comparison". A single shared word is
    not enough: "temperature" appears in a review of every idea in a battery report
    and says nothing about the constraint that happens to contain it.
    """
    shared = wanted & _content_words(text)
    return len(shared) >= 2 and len(shared) / len(wanted) >= 0.4


def _numbered_constraints(indexes: Sequence[int]) -> str:
    """ "constraints one, two and four", against the numbers printed above them."""
    words = [_number_word(index).lower() for index in indexes]
    if len(words) == 1:
        return f"constraint {words[0]}"
    return f"constraints {', '.join(words[:-1])} and {words[-1]}"


def _content_words(text: str) -> set[str]:
    """What a sentence is about, with the grammar and the hedging taken out."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in _RESTATEMENT_STOPWORDS}


_RESTATEMENT_STOPWORDS = _TITLE_MINOR_WORDS | frozenset(
    """lack lacking missing no not none absence absent need needed needs unknown
    unclear whether what how why any some more further additional data evidence
    would could should can may might there""".split()
)


# A sentence about the machinery that wrote the report rather than about the
# field it reports on. The discovery normalizer used to record "No citation-linked
# statements could be normalized." as an uncertainty, and it was printed first
# among seven real open questions -- the one bullet in the section that no
# experiment could ever close. The producer stopped writing it, but a session
# recorded before that still holds it and this is where those are read.
_PIPELINE_DIAGNOSTIC = re.compile(
    r"\bnormaliz|citation-linked statement|\bJSON\b|\bschema\b|"
    r"\bthe (?:extractor|parser|normalizer)\b",
    re.IGNORECASE,
)


def _open_questions(record: ResearchRecord) -> tuple[list[str], str]:
    """What the run left unresolved, minus what the recommendations already listed.

    The meta-review's decision-changing evidence was one of the four sources feeding
    this list, and section nine prints that same list in full a page earlier. Three
    of a live report's four bullets here were its sentences again, word for word, and
    the fourth was the landscape's rewording of one of them. Both places are load-
    bearing -- section nine because obtaining that evidence is the recommendation,
    this section because it is where a reader looks for what is not settled -- so the
    list stays where it is acted on and this section says where it is.
    """
    decisive = [
        _sentence(item)
        for item in (
            record.manifest.evidence_that_would_change_decision
            if record.manifest
            else []
        )
    ]
    questions = [
        _sentence(item)
        for narrative in (record.discovery.narratives if record.discovery else [])
        for item in list(narrative.uncertainties) + list(narrative.disagreements)
        if not _PIPELINE_DIAGNOSTIC.search(item)
    ]
    open_gaps = [
        gap
        for coverage in (record.discovery.coverage_history if record.discovery else [])
        for gap in coverage.gaps
        if gap.status == "open"
    ]
    # A gap against a facet the pass scores is one of a fixed set of seven, and a run
    # that scored zero on all of them printed seven bullets differing by one noun
    # each. Two of the seven were ungrammatical besides, because the facet's enum name
    # went to the page with its underscores swapped for spaces -- "No adequate
    # negative null evidence was discovered." The facets are therefore named from
    # FACET_PHRASES rather than from the recorded description, which on a session
    # written before that mapping existed is the enum spelling.
    scored = [gap.facet for gap in open_gaps if gap.facet in FACET_PHRASES]
    questions.extend(
        _sentence(gap.description)
        for gap in open_gaps
        if gap.facet not in FACET_PHRASES
    )
    if scored:
        named = [
            FACET_PHRASES[facet] for facet in EVIDENCE_FACETS if facet in set(scored)
        ]
        questions.append(
            "The discovery pass found no adequate evidence under "
            + (
                "any facet it scores"
                if len(named) == len(EVIDENCE_FACETS)
                else f"{_plural(len(named), 'facet')} it scores"
            )
            + f": {_names(named)}."
        )
    if record.landscape and record.landscape.coverage_gaps:
        # The clustering fallback carries three coverage gaps of its own -- "Negative
        # and null-result evidence.", "External-validity boundary conditions." -- and
        # set among questions written about this goal they read as findings about it
        # rather than as the template's defaults. They are bare labels besides, where
        # every other bullet here is a stated sentence.
        if any(note.stage == "proximity" for note in record.fallback_stages):
            questions.append(
                "The clustering stage fell back to a template, so what it recorded "
                "as coverage gaps is that template's default list rather than "
                "anything this run found missing, and it is not stated as an open "
                "question here."
            )
        else:
            questions.extend(_sentence(item) for item in record.landscape.coverage_gaps)
    # Ordering the decisive items first makes them win every overlap comparison, so a
    # question that restates one is dropped rather than the other way round.
    deduped = [
        item
        for item in _without_restatements(decisive + questions)
        if item not in decisive
    ]
    lead_in = (
        "The evidence that would change the recommendation is a list of its own, "
        "stated in full under Recommendations and Next Steps above; what follows is "
        "what the run left open besides it."
        if decisive
        else ""
    )
    if deduped:
        return deduped, lead_in
    if decisive:
        return (
            [
                "Nothing was left open beyond the evidence named under "
                "Recommendations and Next Steps above."
            ],
            "",
        )
    return (
        [
            "What primary source would settle the leading claim, and can it be "
            "inspected rather than merely retrieved?"
        ],
        "",
    )


def _shortlist_caveats(
    shortlisted: Sequence[IdeaBrief], briefs: Sequence[IdeaBrief]
) -> list[str]:
    """The two things about a shortlist that only the standings beside it reveal.

    A shortlist reads as a verdict however it is captioned, so where the record shows
    the cut was not a verdict it has to say so. Both cases below were live: the cut
    fell inside a three-way tie on 1184, making the last place on the list a tie-break
    rather than a result, and an idea a reviewer had asked to reject was carried
    forward with nothing on the page reconciling the two.
    """
    caveats = []
    cut = min(brief.elo for brief in shortlisted)
    excluded_at_cut = [
        brief.title for brief in briefs if not brief.shortlisted and brief.elo == cut
    ]
    if excluded_at_cut:
        included_at_cut = [brief.title for brief in shortlisted if brief.elo == cut]
        caveats.append(
            f"The cut fell inside a tie. {_joined_titles(included_at_cut)} made the "
            f"shortlist on {cut}, and "
            + _joined_titles(excluded_at_cut)
            + f" finished on the same {cut} and did not. Nothing in the tournament "
            "separates them: the last "
            + ("place" if len(included_at_cut) == 1 else "places")
            + " on this list came from the order the sort happened to produce, and "
            "the ideas left just outside it should be read as level with the ideas "
            "just inside."
        )
    rejected = [
        brief
        for brief in shortlisted
        if any(review.recommendation == "reject" for review in brief.reviews)
    ]
    if rejected:
        caveats.append(
            _joined_titles([brief.title for brief in rejected])
            + (" was" if len(rejected) == 1 else " were")
            + " shortlisted while carrying a reviewer's recommendation to reject "
            + ("it" if len(rejected) == 1 else "them")
            + ". The shortlist is drawn on tournament rating and does not read the "
            "recommendations, so the two are not in conflict by any rule this run "
            "applied — but nothing reconciled them either, and the rejection stands "
            "unanswered against an idea the run went on to develop."
        )
    return caveats


def _convergence(tournament: TournamentState, briefs: Sequence[IdeaBrief]) -> str:
    """Whether the ordering settled, in units the reader can act on.

    ``score_movement`` is not a score and is not in points: it is the largest
    rating change in the final round as a fraction of the 1200 every idea starts
    on. Printed raw as "a score movement of 0.04 across the final round" it reads
    as four hundredths of a rating point -- a tournament that has all but stopped
    moving -- when the run it came from moved a rating by forty-six points in that
    round, which is wider than most of the gaps in the standings above it. One
    reviewer read the number exactly that way and filed the paragraph's "still
    moving" as a contradiction of its own figure. The fraction is converted to
    points, and the rule it is measured against is stated rather than implied.

    ``ranking_stable_rounds`` counts trailing rounds ending on the same top four
    and floors at one, so "1 round in which the order did not change" describes a
    baseline rather than a finding.

    A movement of exactly ``UNMEASURED_MOVEMENT`` is the sentinel for a tournament
    that recorded no rating change to read, and multiplying it out said the final
    round moved a rating by the whole 1200 an idea starts on -- two sentences after
    the same section reported that no match in the run moved one by more than 16.
    The paragraph then declared every position in the standings provisional on that
    basis. Where nothing was measured the paragraph says so and draws nothing.
    """
    measured = tournament.score_movement < UNMEASURED_MOVEMENT
    points = round(tournament.score_movement * DEFAULT_ELO)
    stable = tournament.ranking_stable_rounds
    rule = (
        "The ranking "
        + ("converged" if tournament.converged else "did not converge")
        + " under the rule this run applies: two consecutive rounds ending on the same "
        "top four, and a final round that moves no rating by more than five per cent "
        f"of the {round(DEFAULT_ELO)} every idea starts on."
    )
    # The counter floors at one, so a run in which no two rounds agreed still counts
    # one. "The top four came out in the same order in the final round only" reported
    # that floor as an observation about the final round, when what it means is that
    # the first half of the rule was never met.
    order = (
        "No two consecutive rounds ended on the same top four, which is where this "
        "run fell short of the first half of that rule."
        if stable <= 1
        else "The top four came out in the same order in the last "
        f"{_plural(stable, 'round')}."
    )
    if not measured:
        return (
            # "recorded no rating change" reads as a change of zero, which would
            # satisfy the second half of the rule rather than leave it untested. The
            # sentinel means the run did not record the movement at all.
            f"{rule} This tournament did not record how far its final round moved the "
            "ratings, so the second half of that rule was not tested here and the "
            f"ordering below rests on the match results alone. {order}"
        )
    opening = (
        f"{rule} The final round moved one rating by "
        f"{tournament.score_movement * 100:.1f} per cent of that, or about "
        f"{_plural(points, 'point')}. {order}"
    )
    if tournament.converged:
        return (
            f"{opening} A converged ranking means the last rounds stopped reordering "
            "the leaders, which is the strongest statement this run can make about the "
            "ordering: it is not a claim that further matches would leave it alone."
        )
    gaps = [abs(first.elo - second.elo) for first, second in pairwise(briefs)]
    narrower = sum(1 for gap in gaps if gap < points)
    # "further than six of the six gaps" is a construction nobody writes, and the
    # reader has to compare two numbers to learn that the answer is all of them.
    noun = _plural(len(gaps), "gap").partition(" ")[2]
    counted = (
        f"the only {noun}"
        if len(gaps) == 1
        else f"all {_number_word(len(gaps)).lower()} {noun}"
        if narrower == len(gaps)
        else f"{_number_word(narrower).lower()} of the "
        f"{_number_word(len(gaps)).lower()} {noun}"
    )
    tail = (
        f"That last round moved a rating further than {counted} between neighbouring "
        "positions in the standings above, so those positions are provisional in the "
        "plainest sense: another round could have swapped them."
        if narrower
        else "No position in the standings above is separated from its neighbour by "
        "less than that, so the run stopping early is not by itself a reason to "
        "distrust the order."
    )
    return f"{opening} The run stopped before the order held still. {tail}"


def _lead_over_rival(leader: IdeaBrief, briefs: Sequence[IdeaBrief]) -> str:
    """How far clear the leader finished, and whether that distance means anything.

    The noise caveat used to print unconditionally, so a 56-point lead was
    followed by "a gap of fewer than roughly thirty points is inside the noise",
    a warning about a gap nothing on the page had. It is stated only about the
    gap it applies to now, and "fewer" is wrong for a rating in any case.

    The thirty was invented. Nothing in the record measures the noise of a
    tournament, and a figure with no derivation behind it was being used to certify
    an ordering as real. What the record does fix is how far one decided match can
    move a rating -- the K factor -- so the gap is stated in those units and the
    reader can see the arithmetic.

    The K factor is the wrong unit to state it in. Elo moves the winner by
    ``K x (1 - expected)``, so K is only reached when a rating the tournament had
    written off wins; between two ideas the run rates as near-equals the move is
    about half of it. Asserting "one decided match moves a rating by at most 32
    points" over rating tables in the same report that show no move above 16 was
    wrong by a factor of two, and the "1.8 times what any single result could
    account for" derived from it was wrong by the same factor. The bound is read
    off the matches the run actually played, which is a figure the reader can check
    against those tables, and the multiple is dropped rather than recomputed: it
    was never more than an assertion dressed as arithmetic.

    A rating also does not imply a result. The leading idea on the live run was
    never paired against three of its seven rivals, so its standing over them is
    an inference from shared opponents. A section that reports the leader's wins
    and losses without saying who it never met invites the reader to take the
    ordering as head to head throughout.
    """
    if len(briefs) < 2:
        return "No other idea was ranked against it."
    gap = abs(leader.elo - briefs[1].elo)
    largest = max(
        (abs(match.swing) for brief in briefs for match in brief.matches),
        default=0,
    )
    parts = [f"Its nearest rival, {briefs[1].title}, ended {gap} points behind."]
    if largest:
        parts.append(
            f"No single match in this tournament moved a rating by more than "
            f"{_plural(largest, 'point')} — the K factor of {round(ELO_K)} is the "
            "most one could ever move it, and only against an idea the ratings had "
            "already written off — so that gap is "
            + (
                "inside what one different result could have produced, and the two "
                "should be read as level rather than ordered."
                if gap <= largest
                else "wider than any single result here produced, and no one match "
                "decides the order between them."
            )
        )
    faced = {match.opponent_title for match in leader.matches}
    unmet = [brief.title for brief in briefs[1:] if brief.title not in faced]
    if unmet:
        parts.append(
            "It was never paired against "
            + _joined_titles(unmet)
            + f", so {_plural(len(unmet), 'idea')} of the "
            f"{_number_word(len(briefs) - 1).lower()} it is "
            "ranked above never met it: those positions rest on shared opponents "
            "rather than on a result between them."
        )
    return " ".join(parts)


def _sentence_of_titles(titles: Sequence[str], *, fallback: str) -> str:
    """A list of titles closed as its own sentence, still capitalised."""
    joined = _joined_titles(titles)
    return _sentence(joined) if joined else fallback


def _joined_titles(titles: Sequence[str], *, fallback: str = "") -> str:
    """A list of idea titles as the subject of a sentence, still capitalised.

    ``_join`` is built for a list of stated sentences: it closes the list with a
    full stop and folds each item after the first down to lower case. Both are
    wrong for titles standing as a subject -- "... Coating Applied. converge on"
    put a stop mid-clause, and "; and an Artificial Cathode-electrolyte Interphase"
    demoted a heading the report prints capitalised everywhere else.
    """
    cleaned = [" ".join(title.split()).rstrip(".") for title in titles if title.strip()]
    if not cleaned:
        return fallback
    if len(cleaned) == 1:
        return cleaned[0]
    # The semicolon is what keeps a list of multi-word titles readable, and it is
    # noise on a list of two: "Dry-coating NCM811 Cathodes; and Protective Al2O3
    # Coatings" punctuates a pair as though more were coming.
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return "; ".join(cleaned[:-1]) + f"; and {cleaned[-1]}"


def _minority_note(record: ResearchRecord, candidate_id: str) -> str:
    """Why a protected idea was protected, checked against where it actually placed.

    The note used to open "it ranks poorly but is the only idea exploring its
    region" for every protected id. In a live run one of them was rank 1 on the
    highest Elo in the tournament, and the report told the reader its winner
    ranked poorly. Protection is about coverage, not about placing: what the note
    can always say is that the region would close without it.
    """
    ranked_id = record.ranked_id(candidate_id)
    title = record.title_for(ranked_id)
    ratings = record.tournament.ratings if record.tournament else {}
    rating = ratings.get(ranked_id)
    below = sum(1 for value in ratings.values() if value < (rating or 0.0))
    placing = (
        "It ranks poorly on its own merits, but it "
        if rating is not None and below < len(ratings) / 2
        else "It "
    )
    # Sole occupancy is a fact about the cluster map, so it is read off the map. A
    # live report called an idea the only one exploring its region two lines after
    # naming the other idea in that same region, which is the report contradicting
    # itself inside one section.
    neighbours = [item for item in record.cluster_of(ranked_id) if item != ranked_id]
    if not neighbours:
        note = (
            f"{title} was protected as a minority hypothesis. {placing}is the only "
            "idea exploring its region of the problem, so discarding it would close "
            "off that region entirely."
        )
    else:
        note = (
            f"{title} was protected as a minority hypothesis. {placing}shares its "
            "region of the problem only with "
            + _joined_titles(
                [record.title_for(item) for item in neighbours],
                fallback="no other idea",
            )
            + ", and the ideas in a region tend to fail together, so dropping this one "
            # "The rest of a thin set" describes several survivors, and a region with
            # two ideas in it has one. The reader was left to work out that "the rest"
            # was the idea named in the same sentence.
            + (
                "would leave the region resting on that idea alone."
                if len(neighbours) == 1
                else "would leave the region resting on the rest of a thin set."
            )
        )
    # Protection and exclusion are separate decisions taken by separate stages, and a
    # live report stated both about the same idea pages apart without connecting them:
    # it read as the report contradicting itself. It is not a contradiction, but the
    # reader has to be told which of the two governs what happens next.
    #
    # Both stages have to be consulted, because they disagree. The sentence used to
    # fire on the reviews alone, so the one idea the two stages disagreed about -- the
    # meta-review excluded it, no reviewer had faulted it -- was the one idea that got
    # no reconciling sentence at all, and the Knowledge Base said it was preserved
    # while section eight said it was excluded.
    faulted = ranked_id in record.recorded_fatal_flaw_ids
    listed = ranked_id in {
        record.ranked_id(item)
        for item in (
            record.manifest.unresolved_fatal_flaw_candidate_ids
            if record.manifest
            else []
        )
    }
    if faulted or listed:
        if faulted and listed:
            authority = (
                "a reviewer recorded a fatal flaw against it and the meta-review "
                "excluded it on that basis"
            )
        elif faulted:
            authority = (
                "a reviewer recorded a fatal flaw against it, which the meta-review "
                "did not act on"
            )
        else:
            authority = (
                "the meta-review excluded it, though no reviewer recorded a fatal "
                "flaw against it"
            )
        note += (
            " It is nonetheless outside any recommendation this report makes: "
            f"{authority}. Protection keeps the region open for a future run; it does "
            "not clear this idea to be acted on."
        )
    return note


@dataclass(frozen=True)
class _ConnectionCounts:
    """How many entries of each kind the cross-link list holds.

    The lead-in used to be handed the converging count and the total, and inferred
    the rest as "not pairs". Two of the three kinds are pairs, so on a run with a
    duplicate pair the lead-in said of it exactly what it was not.
    """

    converging: int = 0
    duplicates: int = 0
    sole_minority: int = 0
    """Protected ideas that are the only occupant of their region."""
    shared_minority: int = 0
    """Protected ideas that share their region with at least one other idea.

    Kept apart from ``sole_minority`` because the lead-in used to describe every
    minority entry as a region held open by one idea, six lines above an entry
    naming the other idea in that same region.
    """
    named_mechanisms: bool = True
    """Whether the clusters carry mechanisms that tell them apart.

    Where they do not, section three says so and prints the cluster names alone. The
    lead-in here went on sending the reader there for "the mechanism its cluster is
    named for", which on that run was one filler sentence copied into all four.
    """

    @property
    def minority(self) -> int:
        return self.sole_minority + self.shared_minority


def connections_lead_in(counts: _ConnectionCounts) -> str:
    """What the heading over the cross-links does not say, said once.

    "Unexpected Connections" is the reference layout's heading and is kept, but on its
    own it claims something the run never established: nothing here surprised anyone,
    and the clustering pass does not report surprise. What it reports is where two
    ideas rest on one mechanism, which the rankings hide -- listed apart, two entries
    look like two bets. Saying so once also takes the consequence out of the bullets,
    where it was printed word for word under every converging pair.
    """
    opening = (
        "Nothing here was flagged as surprising; the run does not judge surprise. "
    )
    if not counts.converging:
        clauses = [
            opening + "What this section reports is what the ideas have in common "
            "that their separate rankings do not show."
        ]
    else:
        clauses = [
            opening
            + "What this section reports is where "
            + ("two ideas rest" if counts.converging == 1 else "ideas rest")
            + (
                " on a single mechanism, which their separate rankings hide: listed "
                "apart they look like separate bets, and they stand or fall on the "
                "same claim. A converging pair is therefore worth less as a portfolio "
                "than its two rankings suggest, and the mechanism its cluster is "
                "named for — stated in full under Main Research Directions above — "
                "is the thing to test first."
                if counts.named_mechanisms
                else " in the same cluster, which their separate rankings hide. What "
                "the clustering has not recorded is any mechanism that tells its "
                "clusters apart, as Main Research Directions above sets out, so a "
                "pair below shares a label and not a demonstrated failure mode: "
                "whether funding both buys two ideas' worth of information is left "
                "open here rather than answered."
            )
        ]
    if counts.duplicates:
        clauses.append(
            "A near-duplicate entry is a stronger finding than a shared mechanism: "
            "those ideas are not two approaches to one question but one approach "
            "stated twice, and funding both buys nothing the first does not."
        )
    if counts.minority:
        # The minority notes are about how thinly one region of the problem is
        # covered, which is the same subject read the other way round. How thinly
        # differs per entry, and a lead-in that generalises over both cases states
        # of one of them the opposite of what its own bullet says.
        cover = []
        if counts.sole_minority:
            cover.append(
                "one where a region rests on a single idea, so dropping it closes "
                "the region rather than narrowing it"
            )
        if counts.shared_minority:
            cover.append(
                "one where a region has more than one occupant but few enough that "
                "they would likely fail together"
            )
        # Singular where there is one of them. "The remaining entries are the inverse
        # case ... They are about how thinly a region is covered" stood over a single
        # bullet, and a reader who counts what a plural promises finds one entry.
        clauses.append(
            (
                "The remaining entry is the inverse case, a protected minority: "
                if counts.minority == 1
                else "The remaining entries are the inverse case, a protected "
                "minority: "
            )
            + _series(cover)
            + ". "
            + ("It is" if counts.minority == 1 else "They are")
            + " about how thinly a region is covered rather than about any one "
            "idea's merits."
        )
    return " ".join(clauses)


def _unexpected_connections(
    record: ResearchRecord,
) -> tuple[list[str], _ConnectionCounts]:
    """Cross-links the ideas share, which is where duplicated effort hides.

    Every list of ideas in the report is in tournament order, and this one was in
    whatever order the clustering pass happened to emit -- so the ideas a reader had
    just met as first and third came back as third and first, reading as a different
    pair. Clusters are ordered by their best-ranked member, members within a cluster
    and titles within a bullet by their own rank.

    Returned with a count of each kind of entry, because what introduces the list
    depends on which kinds are in it.
    """
    ratings = record.tournament.ratings if record.tournament else {}

    def by_rank(candidate_id: str) -> tuple[float, str]:
        return (-ratings.get(record.ranked_id(candidate_id), 0.0), candidate_id)

    def ranked_members(candidate_ids: Sequence[str]) -> list[str]:
        # Resolved to the ranked field first: two revisions of one idea are not two
        # ideas converging, and a cluster naming a revision by its own id would put a
        # title in this sentence that appears nowhere else in the report.
        return sorted(
            dict.fromkeys(record.ranked_id(item) for item in candidate_ids),
            key=by_rank,
        )

    clusters = []
    for cluster in record.landscape.clusters if record.landscape else []:
        members = ranked_members(cluster.candidate_ids)
        if len(members) < 2:
            continue
        # The join closes its list as a sentence, which is right where it stands
        # alone and wrong here: the titles are this sentence's subject, and the
        # full stop landed mid-clause -- "... Coating Applied. converge on one
        # mechanism".
        subject = _joined_titles([record.title_for(item) for item in members])
        clusters.append(
            (
                by_rank(members[0]),
                # The mechanism itself used to be printed here, which made this bullet
                # the third place in the report to state it and the second within a
                # page. What this section adds is not the mechanism -- it is that these
                # particular ideas, ranked apart, stand on it together. Naming the
                # cluster says that, and where the mechanism is stated is in the lead-in
                # above, which is the one place it can be said without saying it once
                # per bullet.
                f"{subject} are the "
                f"{_number_word(len(members)).lower()} ideas in the {cluster.name} "
                "cluster.",
            )
        )
    duplicates = []
    for duplicate in record.landscape.duplicates if record.landscape else []:
        merged = ranked_members(duplicate)
        if len(merged) < 2:
            continue
        duplicates.append(
            (
                by_rank(merged[0]),
                "The following were flagged as near-duplicates and should be merged "
                "before either is funded: "
                + _sentence(
                    _joined_titles([record.title_for(item) for item in merged])
                ),
            )
        )
    protected_ids = sorted(
        dict.fromkeys(
            record.ranked_id(item)
            for item in (
                record.landscape.protected_minority_ids if record.landscape else []
            )
        ),
        key=by_rank,
    )
    minority = [_minority_note(record, protected) for protected in protected_ids]
    # Sole occupancy is what the note branches on, so it is what the lead-in has to
    # be counted on: a protected idea sharing its region is not a region held open
    # by one idea, whatever the heading over the list says.
    sole = sum(1 for item in protected_ids if len(record.cluster_of(item)) < 2)
    connections = (
        [entry for _, entry in sorted(clusters, key=lambda item: item[0])]
        + [entry for _, entry in sorted(duplicates, key=lambda item: item[0])]
        + minority
    )
    return (
        connections
        or [
            "No cross-idea connection was identified beyond the shared framing of "
            "the goal."
        ],
        _ConnectionCounts(
            converging=len(clusters),
            duplicates=len(duplicates),
            sole_minority=sole,
            shared_minority=len(minority) - sole,
            named_mechanisms=_named_mechanisms(
                record.landscape.clusters if record.landscape else []
            ),
        ),
    )


# A Deep Research report tags most of its sentences with the facet the pass was
# sent to cover, and cites by a marker numbered against a source list only that
# pass held. Both are the provider talking to the pipeline. Printed to a reader
# the tag is the same word ninety times over, and the marker sends them to an
# entry of this report's reference list that is a different paper.
_FACET_TAG_RE = re.compile(
    r"[ \t]*\[(?:facet:\s*)?(?:"
    + "|".join(
        sorted(
            (facet.replace("_", "[ _-]") for facet in EVIDENCE_FACETS),
            key=len,
            reverse=True,
        )
    )
    + r")\](?=[\s.,;:)]|$)",
    re.IGNORECASE,
)
_PASS_CITE_RE = re.compile(r"[ \t]*\[cite:[^\]\n]{0,80}\]")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*)$")


def _deep_research_prose(text: str) -> str:
    """One pass's report, as Markdown that sits under this report's own headings.

    What arrives is a whole document: its own title, its own heading tree, a facet
    tag on most sentences and pass-local citation markers. This section used to
    collapse each report's whitespace and run all seven together, which put
    nineteen thousand characters on a single line -- headings inlined as literal
    text -- and, because that line began with a hash, put the whole of it into one
    entry of the table of contents.
    """
    stripped = _PASS_CITE_RE.sub("", _FACET_TAG_RE.sub("", text))
    lines = []
    for line in stripped.splitlines():
        heading = _MARKDOWN_HEADING_RE.match(line.strip())
        if heading:
            # Demoted, not dropped: the pass's own structure is how a fifteen-page
            # literature report stays navigable, and it has to nest under the
            # "### Pass n" heading this section puts above it.
            depth = min(6, 3 + len(heading.group(1)))
            lines.append(f"{'#' * depth} {heading.group(2).strip()}")
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _narrative_facet(narrative: DiscoveryNarrative) -> str:
    """Which facet a pass covered, from the pass itself or from what it returned."""
    if narrative.facet in FACET_PHRASES:
        return narrative.facet
    counted = Counter(
        statement.facet
        for statement in narrative.statements
        if statement.facet in FACET_PHRASES
    )
    return counted.most_common(1)[0][0] if counted else ""


def _knowledge_summary(record: ResearchRecord) -> str:
    narratives = [
        narrative
        for narrative in (record.discovery.narratives if record.discovery else [])
        if narrative.summary.strip()
    ]
    if narratives:
        # The discovery pass writes this in the voice of a settled literature review
        # -- "experimental data strongly supports the mechanism" -- and the report
        # printed it as the opening of its own Knowledge Base. On a run where no
        # source was ever opened, that is the strongest claim in the document and
        # nothing on the page said whose claim it was or what stood behind it.
        checked = any(
            claim.verification_status in {"verified", "corrected"}
            for claim in (record.evidence.claims if record.evidence else [])
        )
        parts = []
        if not checked:
            parts.append(
                "What follows is the literature search's own report of what it "
                "found. No source behind it was opened and checked in this run, so "
                "it is a report of what the search returned rather than a finding "
                "about the field, and the confidence in its wording is the search's "
                "own."
            )
        # A pass that came back with no report text is dropped from this list, and
        # the sentence counted what was left: a live run printed "the search ran as
        # six separate passes" under an appendix saying Deep Research ran seven. The
        # count of passes and the count of reports are two numbers, and where they
        # differ both are stated.
        ran = len(record.discovery.runs) if record.discovery else 0
        silent = max(ran - len(narratives), 0)
        if len(narratives) > 1:
            parts.append(
                f"The search ran as {_number_word(max(ran, len(narratives))).lower()} "
                "separate passes, each asked for a different kind of evidence, and "
                + (
                    "each wrote its own report."
                    if not silent
                    else f"{_number_word(len(narratives)).lower()} of them wrote a "
                    "report. "
                    + (
                        "The other recorded none"
                        if silent == 1
                        else f"The other {_number_word(silent).lower()} recorded none"
                    )
                    + ", and what those passes were asked is under Literature "
                    "discovery in the appendix."
                )
                + " The reports are reproduced below one per pass and in the order "
                "they ran, because a pass that found nothing is a finding about the "
                "literature and disappears when the reports are merged."
            )
        if any(_PASS_CITE_RE.search(item.summary) for item in narratives):
            parts.append(
                "Each pass numbered its citations against its own source list, "
                "which is not the numbering under References in this report, so "
                "those markers are removed here rather than left pointing at the "
                "wrong paper. Which sources a pass found is recorded per pass in "
                "the discovery appendix."
            )
        for index, narrative in enumerate(narratives, start=1):
            # A single pass needs no heading over it: there is nothing to tell it
            # apart from, and "Pass 1" over the only report is a heading that
            # reports the absence of a fan-out.
            if len(narratives) > 1:
                phrase = FACET_PHRASES.get(_narrative_facet(narrative), "")
                number = narrative.pass_number or index
                parts.append(
                    f"### Pass {number}: {phrase[0].upper() + phrase[1:]}"
                    if phrase
                    else f"### Pass {number}"
                )
            parts.append(_deep_research_prose(narrative.summary))
            if narrative.truncated:
                parts.append(
                    "*This pass's report is longer than the run stores in the "
                    "manifest and is cut off above, on the last sentence that "
                    "fitted. The whole of it is in the stored artifact this pass "
                    "recorded.*"
                )
            if not narrative.statements:
                parts.append(
                    "*No statement in this pass could be tied to a source the "
                    "provider also returned, so nothing from it was carried into "
                    "the evidence base or cited elsewhere in this report. It is "
                    "printed here as what the pass reported and nothing more.*"
                )
        return "\n\n".join(parts)
    if _evidence_statements(record):
        # Saying no external knowledge was consulted, on a page that goes on to list
        # four cited papers, was the report contradicting itself. A search-grounded
        # pass returns findings without the synthesis Deep Research writes over
        # them: less than a knowledge base, but not nothing.
        return (
            "Discovery searched the literature and returned individual findings, but no "
            "synthesis across them, so this section cannot state what the field as a "
            "whole holds. "
            # Named here for the same reason section three names it: without the
            # substitution the sentence above reads as a limit of the search, and
            # with it as a limit of this run's configuration.
            + (
                # The clause that closed this sentence sent the reader to the appendix
                # for "what that cost", and the appendix prints a cost only for a Deep
                # Research pass that ran -- which by this branch it did not, so the
                # reader was sent after a figure that is not there. What the appendix
                # does hold about the substitution is already pointed at from Main
                # Research Directions above, so this says the fact and stops.
                "The Deep Research agent, whose output this section is written to "
                "carry, did not run on this goal; a single search-grounded pass "
                "stood in for it. "
                if record.deep_research_stood_in
                else ""
            )
            + "The findings are cited where the narrative uses them and their sources "
            "are listed under References, and they should be read as separate results "
            "rather than as a survey of the field."
        )
    return (
        "Discovery returned no literature summary for this goal. What the workflow "
        "knows is therefore limited to the goal as stated, the ideas it generated from "
        "that goal, and the reviews those ideas received. No external body of knowledge "
        "was consulted, so nothing in this report should be cited as a statement about "
        "the field."
    )


def synthesize_overview(record: ResearchRecord) -> ResearchOverview:
    """Assemble the nine narrative sections from the accepted artifacts.

    This is the derived path. It is deliberately not padded: every sentence traces to
    a field of a validated payload, so an empty artifact produces a shorter and more
    obviously incomplete report rather than a plausible-looking one.
    """
    briefs = build_idea_briefs(record)
    drafts = [
        _section_one(record),
        _section_two(record),
        _section_three(record),
        _section_four(record, briefs),
        _section_five(record, briefs),
        _section_six(record, briefs),
        _section_seven(record, briefs),
        _section_eight(record, briefs),
        _section_nine(record, briefs),
    ]
    directions = _research_directions(record) or (
        # The fallback used to be the cluster table reprinted as bullets, name and
        # mechanism apiece, a few hundred words under the paragraph that had just
        # stated all of them. A reader who has met a mechanism does not need it again
        # under a heading that promises something else: what the clusters are is a
        # description of the ideas that were generated, not a direction drawn from the
        # literature, and this says which it is and where it is.
        [
            "No research direction was drawn from the literature for this run; the "
            "directions the ideas actually took are the clusters they fell into, "
            "named with the mechanism each one shares under Main Research Directions "
            "above."
        ]
        if record.landscape and record.landscape.clusters
        else []
    )
    goal_title = _goal_title(record.session)
    connections, connection_counts = _unexpected_connections(record)
    open_questions, open_questions_lead_in = _open_questions(record)
    return ResearchOverview(
        goal_title=goal_title,
        report_title=f"Ranked Research Ideas{_for_the_goal(record.session)}",
        sections=_fit_word_budget(drafts),
        research_directions=directions
        or ["No research direction was recorded for this run."],
        review_summary=_review_summary(briefs),
        knowledge_summary=_knowledge_summary(record),
        open_questions=open_questions,
        open_questions_lead_in=open_questions_lead_in,
        unexpected_connections=connections,
        connections_lead_in=connections_lead_in(connection_counts),
        source="deterministic_fallback",
    )
