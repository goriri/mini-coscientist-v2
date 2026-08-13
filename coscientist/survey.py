"""Merge every literature pass into one survey the reader can cite from.

The fan-out asks seven questions and gets seven reports, each written as if it
were the only one. Printing them one after another gave the Knowledge Base a
stack of seven literature reviews that repeated each other's background,
disagreed without noticing, and left the merging to the reader -- and because
each pass numbers its own citations, every marker in them had to be struck
before printing, so the longest section of the report was also the only one with
no references in it.

This module does the merge once, with a model that can see all seven reports in
full, and renumbers their citations against the run's own source list on the way
in so what comes back is already cited in the report's terms.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from .evidence import (
    CITE_SPAN,
    EvidenceArtifactStore,
    canonicalize_url,
    renumber_report,
)
from .models import (
    FACET_PHRASES,
    DiscoveryManifest,
    KnowledgeSurvey,
    SourceLead,
)
from .normalization import try_parse_contract

logger = logging.getLogger(__name__)

SYNTHESIS_ROLE = "evidence_synthesis"

REPORT_CHARACTER_BUDGET = 240_000
"""How much pass prose the survey prompt carries, across every pass together.

Seven Deep Research reports run to about thirty-four thousand characters each,
so the whole corpus fits with room to spare and no pass is normally cut. The
budget is here for the run that comes back with far more than that, where the
alternative to trimming is a request the model refuses on length -- and losing
the survey entirely is worse than losing the tail of one pass.
"""

MIN_REPORT_CHARACTERS = 4_000
"""The shortest a pass may be trimmed to, however many passes are sharing the budget."""


class Completer(Protocol):
    def complete(self, *, role: str, prompt: str) -> str: ...


def fold_title(title: str) -> str:
    """The key two records of the same document agree on.

    Discovery stores an annotation's title through ``" ".join(...)[:300]``, so
    matching an annotation back to the lead it produced means normalizing the
    same way. Case is folded on top: the only thing riding on it is which of two
    spellings of one paper's name got recorded first.
    """
    return " ".join(title.split())[:300].casefold()


class SourceIndex:
    """The numbered source list the survey is told to cite, and how to reach it.

    Two joins, tried in that order. A lead's canonical URL is the publisher's
    where locator resolution managed to follow the grounding redirect, and the
    annotation still carries the redirect, so the URL join answers only for the
    leads that were never rewritten. The title join answers for the rest: the
    lead's title *is* the annotation's title, normalized, because that is where
    discovery got it.
    """

    def __init__(self, leads: Sequence[SourceLead]) -> None:
        self.leads = [lead for lead in leads if lead.canonical_url]
        self._token = {lead.id: f"S{index}" for index, lead in enumerate(self.leads, 1)}
        self._by_url: dict[str, SourceLead] = {}
        self._by_title: dict[str, SourceLead] = {}
        for lead in self.leads:
            self._by_url.setdefault(lead.canonical_url, lead)
            if lead.title:
                self._by_title.setdefault(fold_title(lead.title), lead)

    def __len__(self) -> int:
        return len(self.leads)

    @property
    def ids(self) -> list[str]:
        """The lead behind each token, in token order, for the survey to record."""
        return [lead.id for lead in self.leads]

    def token(self, annotation: dict) -> str:
        """``S12`` for the lead this annotation names, or nothing where it was cut."""
        url = annotation.get("url") or annotation.get("uri") or ""
        if isinstance(url, str) and url:
            try:
                lead = self._by_url.get(canonicalize_url(url))
            except ValueError:
                lead = None
            if lead is not None:
                return self._token[lead.id]
        title = annotation.get("title")
        if isinstance(title, str) and title.strip():
            lead = self._by_title.get(fold_title(title))
            if lead is not None:
                return self._token[lead.id]
        return ""

    def listing(self) -> str:
        """The prompt's source list: one line per token, titled and dated."""
        lines = []
        for index, lead in enumerate(self.leads, 1):
            year = f" ({lead.year})" if lead.year else ""
            title = " ".join((lead.title or lead.canonical_url).split())
            lines.append(f"S{index}. {title}{year} -- {lead.canonical_url}")
        return "\n".join(lines)


def _pass_sections(
    manifest: DiscoveryManifest, store: EvidenceArtifactStore, index: SourceIndex
) -> list[str]:
    """Every pass's full report, renumbered, headed by what the pass was asked.

    The manifest keeps only the normalizer's paragraph of each report -- a
    hundred and fifty words against the thirty thousand characters the provider
    wrote -- so summarising from the manifest would be summarising a summary.
    The reports themselves are in the artifact store, and a pass whose artifact
    has gone falls back to its paragraph rather than being dropped, because a
    thin section is closer to the truth than a missing one.
    """
    paragraphs = {
        narrative.pass_number: narrative.summary
        for narrative in manifest.narratives
        if narrative.summary.strip()
    }
    bodies: list[tuple[int, str, str]] = []
    for run in sorted(manifest.runs, key=lambda item: item.pass_number):
        report = ""
        if run.raw_artifact_reference:
            payload = store.get(run.raw_artifact_reference)
            if payload:
                report = renumber_report(payload, index.token)
        if not report:
            report = CITE_SPAN.sub("", paragraphs.get(run.pass_number, "")).strip()
        if not report:
            continue
        facet = FACET_PHRASES.get(run.facet, "")
        heading = f"PASS {run.pass_number}" + (f" -- {facet}" if facet else "")
        bodies.append((run.pass_number, heading, report))
    return [
        f"{heading}\n{body}"
        for _, heading, body in _trimmed(bodies, REPORT_CHARACTER_BUDGET)
    ]


def _trimmed(
    bodies: list[tuple[int, str, str]], budget: int
) -> list[tuple[int, str, str]]:
    """Cut the longest reports back to a shared budget, on a paragraph boundary."""
    total = sum(len(body) for _, _, body in bodies)
    if total <= budget or not bodies:
        return bodies
    allowance = max(budget // len(bodies), MIN_REPORT_CHARACTERS)
    trimmed = []
    for number, heading, body in bodies:
        if len(body) > allowance:
            head = body[:allowance]
            body = head[: head.rfind("\n\n")] if "\n\n" in head else head
            body += "\n\n[This report is longer than the survey request can carry and is cut off here.]"
        trimmed.append((number, heading, body))
    return trimmed


def survey_prompt(
    question: str, sections: Sequence[str], index: SourceIndex, language: str
) -> str:
    """Ask for one survey of the whole corpus, cited against the run's sources."""
    return (
        f"{language}"
        f"Research question: {question}\n\n"
        f"{len(sections)} literature searches were run on this question, each sent "
        "to look for a different kind of evidence. Every report they returned is "
        "below, in full. Write ONE survey of what this literature says -- not "
        f"{len(sections)} summaries, and not a summary of the reports as documents.\n\n"
        "Requirements:\n"
        "- Organize by what the field knows, not by which search found it. A "
        "section is a topic; several passes will feed one section and one pass "
        "will feed several.\n"
        "- Say things a reader can check: quantities with their units, study "
        "designs, populations, effect directions, dates. Prefer the specific "
        "finding to the general statement of it.\n"
        "- Cite every claim you make, with the S-numbers from the source list "
        "below, written as [S12] or [S12, S7] directly after the claim. These are "
        "the numbers the reports below have already been renumbered to, so a claim "
        "you take from a report keeps whatever markers that sentence carries. "
        "Never invent an S-number that is not in the list, and never cite an "
        "S-number a report did not attach to that claim.\n"
        "- Where two reports disagree, say so under `contested` with both sides "
        "and who holds each. Do not average them into one sentence.\n"
        "- Where a search says it looked for something and did not find it, record "
        "it under `not_found`. An absence in this literature is a finding about "
        "the field and it is the first thing lost when reports are merged.\n"
        "- Write the survey's own prose. Do not open a section by naming the pass "
        "it came from, and do not carry over a report's own headings.\n\n"
        f"--- SOURCE LIST ({len(index)} sources) ---\n"
        f"{index.listing()}\n\n"
        "--- SEARCH REPORTS ---\n" + "\n\n".join(sections)
    )


def write_knowledge_survey(
    manifest: DiscoveryManifest,
    provider: Completer,
    *,
    store: EvidenceArtifactStore | None = None,
    language: str = "",
) -> KnowledgeSurvey | None:
    """Merge the passes into one cited survey, or return nothing and leave them.

    Nothing where there is one pass or none -- there is nothing to merge, and the
    single report already reads as one survey -- where no pass left prose to work
    from, or where the model's answer does not satisfy the contract. In every one
    of those cases the Knowledge Base prints the reports one per pass, which is
    what it did before this existed, so a failure here costs the merge and not
    the section.
    """
    index = SourceIndex(manifest.source_leads)
    sections = _pass_sections(manifest, store or EvidenceArtifactStore(), index)
    if len(sections) < 2 or not index:
        return None
    content = provider.complete(
        role=SYNTHESIS_ROLE,
        prompt=survey_prompt(manifest.question, sections, index, language),
    )
    survey = try_parse_contract(content, KnowledgeSurvey)
    if survey is None or not survey.sections:
        logger.warning(
            "The knowledge survey did not satisfy its contract; the Knowledge Base "
            "will reproduce the %d pass reports instead.",
            len(sections),
        )
        return None
    survey.question = manifest.question
    # Written here rather than trusted from the model: it is the list the prompt
    # handed over, and it is the only thing that can turn an [S7] back into a
    # reference number once the manifest has been revised and re-sorted under it.
    survey.sources = index.ids
    return survey
