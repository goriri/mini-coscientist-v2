"""The report is a document, not a dump: these pin its shape and its silences.

Two families of assertion live here. The first is positional — the nine parts of the
reference layout have to appear once each, in order, because a reader navigates by
that order rather than by search. The second is negational — every construct that
betrays a serialised artifact rather than a written report (fences, braces, raw
hashes, opaque ids, redirect URLs, ``N/A`` stubs) has to be absent from the body.
Absence is the harder property to keep, so it is the one asserted most literally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from types import SimpleNamespace

import pytest

from coscientist.advisories import AUTO_APPROVAL_WARNING
from coscientist.dossier import (
    CHAPTER_SECTIONS,
    SUMMARY_TABLE_HEADING,
    _cell,
    _match_summary,
    _verdict_line,
    compile_dossier,
    shared_match_notes,
    shared_review_tally,
)
from coscientist.models import (
    ApprovalProfile,
    DecisionAction,
    DiscoveryStatement,
    DossierManifest,
    HumanDecision,
    Session,
    SourceLead,
)
from coscientist.narrative import (
    CITATION_ANNOTATION_CEILING,
    CITATION_ANNOTATIONS,
    IDEA_TABLE_ROWS,
    NARRATIVE_WORD_CEILING,
    NARRATIVE_WORD_FLOOR,
    SUMMARY_SUBSECTIONS,
    CitationRegistry,
    _attributed_responses,
    _idea_description,
    _review_answer,
    _section_eight,
    _section_four,
    build_idea_briefs,
    derive_idea_title,
    idea_title,
    load_record,
    support_notice,
    synthesize_overview,
    unique_titles,
)
from coscientist.orchestration import CoScientistWorkflow

# Everything from this heading on is the appendix half: the warnings chapter, then
# the audit trail where ids, schema names and payload internals are the point. The
# anti-pattern rules apply to the body above it.
_APPENDIX = "\n# Warnings and Limitations"
_PROVENANCE = "\n# Provenance"


@pytest.fixture
def report(rich_session: Session) -> str:
    return compile_dossier(rich_session)


@pytest.fixture
def body(report: str) -> str:
    return report.split(_APPENDIX)[0]


def _idea_sections(body: str) -> list[str]:
    """The per-idea sections of the deep-dive chapter, and nothing else.

    The chapter opens on two sections of its own -- the summary table and the
    reading guide -- which are level two beside the ideas rather than level three
    under them, so a bare split on the heading level returns them as ideas.
    """
    chapter = body[body.rindex("\n# Top ideas in detail\n") :]
    return [
        section
        for section in chapter.split("\n## ")[1:]
        if section.split("\n", 1)[0].strip() not in CHAPTER_SECTIONS
    ]


def _chapter_preamble(body: str) -> str:
    """Everything in the deep-dive chapter above the first idea."""
    chapter = body[body.index("\n# Top ideas in detail\n") :]
    parts = chapter.split("\n## ")
    kept = [parts[0]]
    for section in parts[1:]:
        if section.split("\n", 1)[0].strip() not in CHAPTER_SECTIONS:
            break
        kept.append(section)
    return "\n## ".join(kept)


def _headings(text: str) -> list[tuple[int, str]]:
    return [
        (len(match.group(1)), match.group(2).strip())
        for match in re.finditer(r"^(#{1,6}) (.+)$", text, re.MULTILINE)
    ]


# --------------------------------------------------------------------------- shape


def test_the_nine_parts_appear_once_each_and_in_the_reference_order(body: str):
    """Part order is the report's table of contents; a reader relies on it."""
    ordered = [
        "# Research Goal Details",
        "## Goal",
        "## Requirements",
        "## Attributes",
        "## Criteria",
        "# Research Overview",
        "## Top ideas",
        "#### 1. Research Goal",
        "## Research directions",
        "## Review summary",
        "# Knowledge Base",
        "## Knowledge Summary",
        "## Open Questions",
        "## Unexpected Connections",
        "## References",
    ]
    positions = []
    for heading in ordered:
        marker = f"\n{heading}\n"
        assert marker in body, f"missing part: {heading}"
        positions.append(body.index(marker))
    assert positions == sorted(positions), "the nine parts are out of order"


def test_the_goal_title_opens_the_document(report: str):
    level, text = _headings(report)[0]
    assert level == 1
    assert text
    assert not text.startswith("Research Goal")


def test_the_narrative_has_exactly_nine_numbered_sections_in_order(body: str):
    numbers = [
        int(match.group(1))
        for match in re.finditer(r"^#### (\d+)\. ", body, re.MULTILINE)
    ]
    assert numbers == list(range(1, 10))


def test_the_attribution_line_matches_the_reference_stamp(body: str):
    stamp = re.search(
        r"^Prepared by \U0001f9ec  AI co-scientist on (\d{4}-\d{2}-\d{2})\. "
        r"For research purposes only\.$",
        body,
        re.MULTILINE,
    )
    assert stamp is not None, "the attribution line is missing or misspaced"


def test_the_attribution_follows_the_report_title_under_top_ideas(body: str):
    overview = body[body.index("\n# Research Overview\n") :]
    heads = _headings(overview)[:4]
    assert [level for level, _ in heads] == [1, 2, 3, 4]
    assert heads[1][1] == "Top ideas"
    assert overview.index("Prepared by") > overview.index(f"### {heads[2][1]}")


def test_the_narrative_lands_inside_the_declared_word_band(rich_session: Session):
    overview = synthesize_overview(load_record(rich_session))
    assert NARRATIVE_WORD_FLOOR <= overview.word_count <= NARRATIVE_WORD_CEILING


def _reference_entries(body: str) -> list[str]:
    section = body[body.index("\n## References\n") :].splitlines()[2:]
    entries = []
    for line in section:
        if line.startswith("#"):
            break
        # Numbered entries only. The list may be introduced by prose stating what
        # holds of every source in it, which is not one of them.
        if re.match(r"\d+\. ", line):
            entries.append(line)
    return entries


def test_references_are_numbered_to_match_their_markers(body: str):
    """An unnumbered list left "[2]" in the prose pointing at nothing."""
    entries = _reference_entries(body)
    assert entries
    for position, entry in enumerate(entries, start=1):
        assert not entry.startswith(("-", "*", "#", "|"))
        assert entry.startswith(f"{position}. "), f"out of sequence: {entry}"


def test_a_reference_with_no_document_locator_says_so_instead_of_linking(body: str):
    section = body[body.index("\n## References\n") :]
    for entry in _reference_entries(body):
        assert "grounding-api-redirect" not in entry
        if "http" in entry:
            continue
        if entry.rstrip().endswith("."):
            # A search redirect leaves neither a title nor a host, so the entry is
            # the publisher and nothing else. Why that is so is stated once above
            # the list rather than on each of the five entries that share it.
            if "Untitled source on " in entry:
                assert "returned a redirect rather than the document" in section
                continue
        # Which of the two it is matters: a front page tells the reader where to
        # look for the paper, nothing at all tells them to search by title.
        assert entry.endswith(
            "the literature search recorded no link to the document itself."
        ) or entry.endswith("so it has to be found by title.")


def test_a_reference_known_only_by_its_publisher_names_the_publisher():
    """ "No resolvable locator was recorded" told the reader not to go looking."""
    from coscientist.dossier import _reference_line
    from coscientist.narrative import Citation

    front_page = _reference_line(
        Citation(number=1, title="A Paper", url="https://www.frontiersin.org/"),
        mark_standing=False,
    )
    assert front_page == (
        "1. A Paper. Retrieved from frontiersin.org; the literature search recorded "
        "no link to the document itself."
    )
    assert _reference_line(
        Citation(number=2, title="A Paper", url=""), mark_standing=False
    ).endswith("No link to this source was recorded, so it has to be found by title.")
    # "It has to be found by title" over an entry that has no title is advice the
    # entry refutes: what the search returned for these is one of its own redirects,
    # which names neither the document nor a host.
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ"
    assert (
        _reference_line(
            Citation(number=3, title="Untitled source on nih.gov", url=redirect),
            mark_standing=False,
        )
        == "3. Untitled source on nih.gov."
    )


def test_an_entry_the_run_could_not_retrieve_says_so_where_the_reader_meets_it():
    """Two lead-in sentences promised "which is which is recorded against each entry
    in the evidence appendix". No entry recorded it, and the appendix that name points
    at lists the ideas whose grounding is in doubt and no entry of this list at all."""
    from coscientist.dossier import _reference_lines
    from coscientist.narrative import Citation

    lines = _reference_lines(
        [
            Citation(
                number=1,
                title="A checked paper",
                url="https://example.org/a.pdf",
                verification_status="verified",
            ),
            Citation(
                number=2,
                title="A lead nobody read",
                url="https://example.org/b.pdf",
                verification_status="discovered_unverified",
            ),
            Citation(
                number=3,
                title="A paper that went away",
                url="https://example.org/c.pdf",
                verification_status="inaccessible",
            ),
        ]
    )

    assert lines[0].endswith("(https://example.org/a.pdf)"), (
        "a mark printed against every entry is not a mark; the checked one is silent"
    )
    assert lines[1].endswith(
        "Not retrieved: this entry records where a statement came from, not that the "
        "document says it."
    )
    assert "Nothing here is grounded by it." in lines[2]


def test_a_list_where_nothing_was_retrieved_says_it_once_over_the_list():
    """A fact true of every entry belongs in the prose that introduces them."""
    from coscientist.dossier import _reference_lines
    from coscientist.narrative import Citation

    lines = _reference_lines(
        [
            Citation(number=n, title=f"Lead {n}", url=f"https://example.org/{n}.pdf")
            for n in (1, 2, 3)
        ]
    )

    assert not any("Not retrieved" in line for line in lines)


def test_two_references_no_title_tells_apart_are_marked_as_separate_records():
    """A live list carried "Untitled source on nih.gov" at 5 and again at 7.

    Neither entry had a link, so a reader meeting [5] and [7] in the text had no
    way to tell whether one source had been numbered twice. An entry that does
    carry its own link is already told apart by the link, and the clause would be
    noise on it.
    """
    from coscientist.dossier import _reference_lines
    from coscientist.narrative import Citation

    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ"
    lines = _reference_lines(
        [
            Citation(number=1, title="Untitled source on nih.gov", url=redirect + "1"),
            Citation(
                number=2,
                title="Untitled source on mdpi.com",
                url="https://www.mdpi.com/1996-1073/17/19/4970",
            ),
            Citation(number=3, title="Untitled source on nih.gov", url=redirect + "2"),
            Citation(
                number=4,
                title="Untitled source on mdpi.com",
                url="https://www.mdpi.com/2073-4360/18/4/429",
            ),
        ]
    )

    assert lines[0] == (
        "1. Untitled source on nih.gov, the first of two separate records the search "
        "returned under that publisher without a title."
    )
    assert lines[2].startswith("3. Untitled source on nih.gov, the second of two")
    assert "separate records" not in lines[1] + lines[3]


def test_a_locator_long_enough_to_wrap_is_shown_short_and_linked_whole():
    """One ResearchGate URL carried the paper's title and ran to three hundred
    characters, which took four lines of the reference list and wrapped mid-word in
    the PDF. Shortening the href instead would have broken the link."""
    from coscientist.dossier import _reference_line
    from coscientist.narrative import Citation

    url = "https://www.researchgate.net/publication/317195885_" + "Long_Title_" * 20
    line = _reference_line(Citation(number=1, title="A Paper", url=url))

    assert f"]({url})" in line
    shown = line[line.index("[") + 1 : line.index("](")]
    assert shown.endswith("…")
    assert len(shown) < 100
    assert url.startswith(shown[:-1])


# ------------------------------------------------------------------ anti-patterns


def test_the_body_carries_no_serialised_artifact_debris(body: str):
    assert "```" not in body, "a code fence reached the report body"
    assert "{" not in body and "}" not in body, "a JSON payload reached the body"
    assert "http" not in body, "a URL reached the body"
    assert not re.search(r"\bN/?A\b", body), "an unfilled stub reached the body"
    assert not re.search(r"[0-9a-f]{12,}", body), "an opaque id reached the body"
    _assert_no_record_ids(body)


def test_the_body_speaks_the_readers_vocabulary_and_not_the_pipelines(body: str):
    """Words that name a part of this system rather than a part of the science.

    Every one of these reached a live report: "Candidate proposes that ZnO acts as a
    chemical HF scavenger" opened a review, "no evidence available in the provided
    packet" closed one, and the auto-approval warning -- the sentence a reader most
    needs to act on -- said the payload had satisfied its contract.
    """
    for word in (
        "packet",
        "payload",
        "artifact",
        "contract",
        "pipeline",
        "schema",
        "enum",
    ):
        found = re.findall(rf"(?i)\b{word}s?\b", body)
        assert not found, f"pipeline vocabulary reached the body: {word}"
    assert not re.search(r"(?:^|(?<=[.!?] ))Candidate [a-z]", body, re.MULTILINE), (
        "a review names its subject 'Candidate' rather than naming the idea"
    )


def test_a_reviewer_is_named_in_words_rather_than_by_its_id(body: str):
    """ "Ethics safety governance review:" is an enum with the underscores taken out."""
    # Either where the review is, or -- once every idea's reviews ask the same
    # questions -- in the one list of who asks what, above the ideas.
    lead_ins = re.findall(
        r"^(?:- \*\*[A-Za-z ]+\*\* — )?([A-Z][A-Za-z, ]+ reviewer?)[.:]",
        body,
        re.MULTILINE,
    )
    assert lead_ins
    for lead_in in lead_ins:
        assert lead_in.endswith(" reviewer"), (
            f"the pass is named, not the reviewer: {lead_in}"
        )
        words = lead_in.removesuffix(" reviewer").split()
        # Three bare nouns in a row is an id; a name has its conjunction.
        assert len(words) < 3 or "and" in words, f"unwritten reviewer name: {lead_in}"


def test_a_question_every_review_asks_of_every_idea_is_asked_once(
    rich_session: Session, body: str
):
    """A review's question is a property of the review, not of the idea under it. Five
    role names and five questions reprinted under each of eight ideas is forty fixed
    paragraphs standing between the reader and the findings that differ."""
    briefs = build_idea_briefs(load_record(rich_session))
    asked = {review.question for brief in briefs for review in brief.reviews}
    assert len(briefs) > 1 and asked

    for question in asked:
        assert body.count(question) == 1, f"asked under every idea: {question}"
    hoisted = _chapter_preamble(body)
    for question in asked:
        assert question in hoisted, "hoisted above the ideas, not dropped"
    assert "Who asks what is set out here rather than under each idea" in hoisted


def test_a_heading_that_claims_more_than_its_content_is_qualified_beneath_it(
    body: str,
):
    """Two headings of the reference layout say something this run cannot support.

    "Unexpected Connections" sat over the clustering result, which reports where two
    ideas share a mechanism and says nothing about surprise. "Identified issues &
    Validated Risks" sat over the risks the proposing specialist wrote about its own
    idea, which no reviewer ever saw. The headings belong to the layout being mimicked
    and stay; what the reader is owed is the qualification underneath them.
    """
    connections = body[body.index("\n## Unexpected Connections\n") :].split("\n## ")[1]
    assert "the run does not judge surprise" in connections, (
        "the connections heading claims surprise the run never assessed"
    )
    assert body.count("Identified issues & Validated Risks"), (
        "the risks subsection is missing"
    )
    # Stated once, above the deep dives, rather than under each of them.
    assert body.count("No reviewer was asked to confirm them") == 1, (
        "a risk list is printed as validated when nothing validated it"
    )


def _assert_no_record_ids(body: str) -> None:
    """No internal id in the prose, including inside a specialist's own sentence.

    The one place an id is allowed is the grounding warning, which has to name the
    exact id it is complaining about for the warning to be actionable. So is an id
    the run cannot resolve, wherever it is set in code font: there is nothing to name
    it after, and describing it instead printed the same phrase once per id inside a
    single pair of brackets.
    """
    warning = "Warning: this idea cites evidence that does not exist"
    prose = "\n".join(
        line for line in body.splitlines() if not line.startswith(warning)
    )
    prose = re.sub(r"`[^`\n]*`", "", prose)
    leaked = re.findall(
        r"\b(?:candidate|cand|claim|source|src|review|rev|hypothesis|lead|stmt"
        r"|statement)[0-9]*_\w+\b",
        prose,
    )
    assert not leaked, f"internal record ids reached the body: {sorted(set(leaked))}"


def test_every_hash_in_the_body_is_a_heading(body: str):
    """A stray ``#`` means markdown leaked as text rather than rendering.

    Link destinations are exempt: the contents list and the index of exhibits
    are made of ``[text](#anchor)``, where the hash is the fragment separator
    and never reaches the reader.
    """
    for raw in body.splitlines():
        line = re.sub(r"\]\(#[^)\s]*\)", "]()", raw)
        if "#" in line:
            assert re.match(r"^#{1,6} \S", line), f"raw hash in prose: {raw!r}"


def test_the_body_never_prints_a_schema_name(body: str):
    for token in (
        "CandidatePopulation",
        "ReviewSet",
        "TournamentState",
        "EvidencePacket",
    ):
        assert token not in body


# --------------------------------------------------------------------- citations


def test_citation_numbers_are_dense_from_one(body: str):
    numbers = sorted(
        {
            int(number)
            for group in re.findall(r"\[(\d+(?:, \d+)*)\]", body)
            for number in group.split(", ")
        }
    )
    assert numbers, "the narrative cites nothing at all"
    assert numbers == list(range(1, len(numbers) + 1))


def test_every_cited_number_resolves_to_a_reference(body: str):
    cited = {
        int(number)
        for group in re.findall(r"\[(\d+(?:, \d+)*)\]", body)
        for number in group.split(", ")
    }
    assert max(cited) <= len(_reference_entries(body))


def test_annotations_come_from_the_closed_set_and_stay_sparse(body: str):
    annotated = re.findall(r"\[\d+(?:, \d+)*\] \(([^)]+)\)", body)
    groups = re.findall(r"\[\d+(?:, \d+)*\]", body)
    assert set(annotated) <= set(CITATION_ANNOTATIONS)
    assert len(annotated) <= CITATION_ANNOTATION_CEILING * len(groups)


def test_annotations_never_appear_in_the_per_idea_deep_dives(body: str):
    deep_dives = body[body.rindex("\n# Top ideas in detail\n") :]
    for annotation in CITATION_ANNOTATIONS:
        assert f"({annotation})" not in deep_dives


def test_the_registry_numbers_by_first_use_not_by_manifest_position():
    leads = [
        SourceLead(canonical_url=f"https://x/{n}", title=f"Lead {n}") for n in range(4)
    ]
    registry = CitationRegistry(leads)
    assert registry.marker(["https://x/3"]) == "[1]"
    assert registry.marker(["https://x/1", "https://x/3"]) == "[1, 2]"
    assert registry.marker(["https://x/unknown"]) == ""
    assert [citation.title for citation in registry.references()] == [
        "Lead 3",
        "Lead 1",
    ]


def test_the_registry_withholds_qualifiers_once_the_page_is_dense():
    leads = [
        SourceLead(canonical_url=f"https://x/{n}", title=f"Lead {n}") for n in range(12)
    ]
    # Eleven of the twelve, not all twelve: a qualifier that holds of every source is
    # withheld outright by the rule below, which would leave nothing for the cap to cap.
    registry = CitationRegistry(
        leads, annotations={f"https://x/{n}": "disputed" for n in range(11)}
    )
    markers = [registry.marker([f"https://x/{n}"]) for n in range(12)]
    # An uncapped renderer would qualify eleven of these. The cap holds the page to a
    # quarter of its citation groups.
    assert registry.annotation_rate <= CITATION_ANNOTATION_CEILING
    assert sum("(disputed)" in marker for marker in markers) == 2
    assert markers[0] == "[1]", "the first citation is never the annotated one"


def test_a_qualifier_true_of_every_source_is_not_printed_beside_any_of_them():
    """A tag says "this source is unlike the others", so it cannot say "all of them".

    On a run where the evidence gate was waived, every source is unverified and the
    per-source rule derives "unsupported" for all of them. Printing that on the
    quarter of citations the ceiling admits tells a reader the other three quarters
    were checked, which is the opposite of what the record holds.
    """
    leads = [
        SourceLead(canonical_url=f"https://x/{n}", title=f"Lead {n}") for n in range(8)
    ]
    registry = CitationRegistry(
        leads, annotations={f"https://x/{n}": "unsupported" for n in range(8)}
    )
    markers = [registry.marker([f"https://x/{n}"]) for n in range(8)]
    assert not any("(" in marker for marker in markers)
    assert registry.annotation_rate == 0.0
    assert registry.universal_qualifier == "unsupported"


def test_a_run_that_recorded_no_verdict_at_all_still_says_nothing_was_checked():
    """No verification record and a recorded "unsupported" are one fact, not two.

    A run whose evidence packet carries no claims annotates no source, so the set of
    qualifiers found was the empty string alone -- which the uniformity rule read as
    "no qualifier holds of every source" and the reference list then said nothing at
    all about whether its sources had been checked. Nothing had been.
    """
    leads = [
        SourceLead(canonical_url=f"https://x/{n}", title=f"Lead {n}") for n in range(4)
    ]
    assert CitationRegistry(leads).universal_qualifier == "unsupported"
    # A partial record is the same fact twice over, so it resolves the same way.
    mixed = CitationRegistry(leads, annotations={"https://x/0": "unsupported"})
    assert mixed.universal_qualifier == "unsupported"
    assert mixed.marker(["https://x/0"]) == "[1]"
    # A qualifier that genuinely distinguishes one source from the rest still does.
    # The annotated source is cited last so the density cap has room to admit it.
    checked = CitationRegistry(leads, annotations={"https://x/0": "leaning accurate"})
    assert checked.universal_qualifier == ""
    markers = [checked.marker([f"https://x/{n}"]) for n in (1, 2, 3, 0)]
    assert markers[-1] == "[4] (leaning accurate)"


def test_one_document_returned_under_two_links_is_one_reference():
    """A live list ran "5. Limitations of Ultrathin Al2O3 Coatings on LNMO Cathodes -
    Diva Portal. The literature search recorded only its own redirect link for this
    source, which no longer resolves" directly above "6. Limitations of Ultrathin
    Al2O3 Coatings on LNMO Cathodes (2021)" with a link that does."""
    title = "Limitations of Ultrathin Al2O3 Coatings on LNMO Cathodes"
    redirect = "https://vertexaisearch.grounding-api-redirect/abc"
    resolved = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8603187/"
    registry = CitationRegistry(
        [
            SourceLead(canonical_url=redirect, title=title),
            SourceLead(canonical_url=resolved, title=title),
        ]
    )

    assert registry.marker([redirect]) == "[1]"
    assert registry.marker([resolved]) == "[1]", "the same paper, so the same number"
    assert [citation.url for citation in registry.references()] == [resolved]


def test_a_repository_appended_to_a_title_does_not_make_it_a_second_paper():
    """The fold above matched whole titles, so the live pair survived it after all.

    One lead carried the paper's name; the other carried the same name with the
    repository's appended, which is not a hostname and so outlives the search-chrome
    cut. They were numbered 5 and 6 with opposite retrieval verdicts, and the
    discredited-grounding appendix then named the same paper twice.
    """
    title = "Limitations of Ultrathin Al2O3 Coatings on LNMO Cathodes"
    redirect = "https://vertexaisearch.grounding-api-redirect/abc"
    resolved = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8603187/"
    registry = CitationRegistry(
        [
            SourceLead(canonical_url=redirect, title=f"{title} - Diva Portal"),
            SourceLead(canonical_url=resolved, title=title, year="2021"),
        ]
    )

    assert registry.marker([redirect]) == "[1]"
    assert registry.marker([resolved]) == "[1]"
    assert registry.folded_duplicates == 1
    assert [citation.url for citation in registry.references()] == [resolved]


def test_two_papers_sharing_no_title_are_not_folded_by_the_repository_rule():
    """The rule folds a suffix onto a title the run already holds, and only then."""
    registry = CitationRegistry(
        [
            SourceLead(canonical_url="https://a/1", title="Coatings on LNMO Cathodes"),
            SourceLead(
                canonical_url="https://a/2", title="Coatings on NMC Cathodes - Elsevier"
            ),
        ]
    )

    assert len(registry.references()) == 0
    assert registry.marker(["https://a/1"]) == "[1]"
    assert registry.marker(["https://a/2"]) == "[2]"
    assert registry.folded_duplicates == 0


def test_a_source_qualified_once_is_qualified_at_every_later_citation_of_it():
    """A live chapter printed "[5] (inaccurate)" on one finding and cited the same
    unretrievable source four more times without it, which reads as four sound
    citations and one bad one rather than five citations of one bad source."""
    leads = [
        SourceLead(canonical_url=f"https://x/{n}", title=f"Lead {n}") for n in range(12)
    ]
    registry = CitationRegistry(leads, annotations={"https://x/0": "inaccurate"})
    # Cited late enough for the density cap to admit it, then four times over.
    markers = [registry.marker([f"https://x/{n}"]) for n in (1, 2, 3, 4, 0, 0, 0, 0)]

    assert markers[4] == "[5] (inaccurate)"
    assert markers[5:] == ["[5] (inaccurate)"] * 3, (
        "the cap decides which sources speak, not which mentions of one source do"
    )


def test_a_source_nobody_could_open_is_not_marked_as_a_finding_that_is_false():
    """Fifteen findings on a live run carried "(inaccurate)" beside the marker while
    their own reference entries, five hundred lines below, read "Could not be
    retrieved when this run went back to it" -- the marker calling the finding false
    and the entry saying only that nobody could open the page."""
    from coscientist.models import EvidenceClaim, EvidencePacket, SourceRecord
    from coscientist.narrative import _claim_annotations

    def _annotated(status: str) -> str | None:
        return _claim_annotations(
            EvidencePacket(
                question="Can a coating help?",
                sources=[
                    SourceRecord(id="s1", url="https://example.org/1", title="Paper")
                ],
                claims=[
                    EvidenceClaim(
                        id="c1",
                        claim="A coating raises retention.",
                        source_id="s1",
                        verification_status=status,
                        confidence=0.9,
                    )
                ],
            )
        ).get("https://example.org/1")

    assert _annotated("retracted") == "inaccurate"
    assert _annotated("inaccessible") == "unsupported"
    assert _annotated("discovered_unverified") == "unsupported"
    # A confident verified check earns no qualifier at all, which is the point of
    # them: they mark the exceptions.
    assert _annotated("verified") is None


def test_two_sources_the_search_left_untitled_are_not_folded_into_one():
    """ "Untitled source on mdpi.com" names a publisher, not a document."""
    registry = CitationRegistry(
        [
            SourceLead(canonical_url="https://www.mdpi.com/a", title=""),
            SourceLead(canonical_url="https://www.mdpi.com/b", title=""),
        ]
    )

    assert registry.marker(["https://www.mdpi.com/a"]) == "[1]"
    assert registry.marker(["https://www.mdpi.com/b"]) == "[2]"


def test_the_head_of_the_reference_list_counts_the_list_below_it():
    """ "Fifteen of the fifty-nine were retrieved and checked" stood at the head of a
    list holding six entries, the corpus figure printed over the cited entries."""
    from coscientist.narrative import ResearchRecord, _cited_reference_standing

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://x/{n}",
                title=f"Lead {n}",
                verification_status="verified" if n < 3 else "discovered_unverified",
            )
            for n in range(9)
        ]
    )
    for n in (0, 4, 5):
        record.citations.number(f"https://x/{n}")

    said = _cited_reference_standing(record)

    assert said.startswith(
        "Of three entries below, one was retrieved and checked against the document "
        "it names, and two record where a statement came from and no more."
    )


def test_a_list_holding_a_checked_entry_does_not_say_none_of_them_was_checked():
    """The blanket sentence is read off claim support and speaks about retrieval.

    On a live run the two records disagreed: twenty-four entries of which 1, 12 and
    20 had been retrieved and checked -- which is why those three alone printed no
    standing under them -- stood beneath "None of the sources below was checked
    against the document it names." A source no claim was annotated against reads as
    unsupported, and verification had read it.
    """
    from coscientist.dossier import _uniform_reference_standing
    from coscientist.narrative import ResearchRecord, _cited_reference_standing

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://x/{n}",
                title=f"Lead {n}",
                verification_status="verified" if n == 0 else "discovered_unverified",
            )
            for n in range(3)
        ]
    )
    for n in range(3):
        record.citations.number(f"https://x/{n}")

    # The qualifier still holds of the claims; it is the sentence that does not hold.
    assert record.citations.universal_qualifier == "unsupported"
    assert _uniform_reference_standing(record) == ""
    # What stands in its place counts, and its count matches the entries' own marks.
    assert _cited_reference_standing(record).startswith(
        "Of three entries below, one was retrieved and checked against the document "
        "it names, and two record where a statement came from and no more."
    )


def test_a_list_where_nothing_was_checked_keeps_the_blanket_sentence():
    from coscientist.dossier import _uniform_reference_standing
    from coscientist.narrative import ResearchRecord

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.citations = CitationRegistry(
        [
            SourceLead(canonical_url=f"https://x/{n}", title=f"Lead {n}")
            for n in range(3)
        ]
    )
    for n in range(3):
        record.citations.number(f"https://x/{n}")

    assert _uniform_reference_standing(record).startswith(
        "None of the sources below was checked against the document it names."
    )


def test_a_reference_list_whose_entries_were_all_checked_says_so():
    from coscientist.narrative import ResearchRecord, _cited_reference_standing

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://x/{n}",
                title=f"Lead {n}",
                verification_status="verified",
            )
            for n in range(2)
        ]
    )
    for n in range(2):
        record.citations.number(f"https://x/{n}")

    # "All two entries below" counts a pair the way nothing else in the report does.
    assert _cited_reference_standing(record) == (
        "Both entries below were retrieved and checked against the document they name."
    )


def test_an_entry_that_failed_verification_is_not_counted_as_one_nobody_checked():
    """ "The other five entries record where a statement came from and no more" stood
    over five entries of which one link reached no document and one document had been
    retracted. Both had been looked at, and both are worse standing than a lead nobody
    has got to yet -- reported as verification not yet attempted."""
    from coscientist.narrative import ResearchRecord, _cited_reference_standing

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    statuses = [
        "verified",
        "verified",
        "discovered_unverified",
        "metadata_verified",
        "corrected",
        "inaccessible",
        "retracted",
    ]
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://x/{n}",
                title=f"Lead {n}",
                verification_status=status,
            )
            for n, status in enumerate(statuses)
        ]
    )
    for n in range(len(statuses)):
        record.citations.number(f"https://x/{n}")

    said = _cited_reference_standing(record)

    # Five standings, not four. The entry whose own line reads "Not checked against
    # the document: only its catalogue record was reached" was counted here under
    # "records where a statement came from and no more", which says nobody looked it
    # up: somebody did, and got as far as the catalogue.
    assert said.startswith(
        "Of seven entries below, three were retrieved and checked against the "
        "document they name, one was found in a catalogue and not read, one records "
        "where a statement came from and no more, one was looked for and could not "
        "be retrieved at all, and one names a document that has since been retracted."
    )
    assert "the other four entries record" not in said


# -------------------------------------------------------------------- idea titles


def test_a_title_is_derived_from_the_claim_rather_than_from_the_id():
    title = derive_idea_title(
        "Test whether a conformal alumina coating suppresses electrolyte "
        "decomposition at the silicon anode surface because the interphase thins"
    )
    assert title.startswith("Conformal Alumina Coating Suppresses")
    assert "because" not in title
    assert not title.lower().startswith("test")


def test_a_truncated_title_never_ends_mid_phrase():
    for claim in (
        "Evaluate the adjacent system and test whether the binder chemistry "
        "changes the observed flux distribution across the electrode",
        "Compare the coating against the untreated control in order to establish "
        "the size of the effect",
    ):
        title = derive_idea_title(claim)
        assert title.split()[-1].lower() not in {"and", "the", "of", "to", "test"}


def test_a_truncated_title_does_not_end_on_a_participle_left_without_its_noun():
    """ "... Observed in Coated" was "in coated electrodes" before the cut.

    Trimming back to "Observed" was itself not far enough: a live governance block
    was headed "Withdrawn: The >10% Extension in Cycle Life Observed", where the
    participle needs a complement the cut removed and reads as an unfinished word.
    """
    title = derive_idea_title(
        "The >10% extension in cycle life observed in coated electrodes is "
        "caused by suppressed gas evolution rather than by the coating"
    )
    assert title == "The >10% Extension in Cycle Life"


def test_a_transitive_verb_left_with_a_one_word_object_gives_the_object_back():
    """ "Coating Extends Cycle" named a coating that extends a cycle."""
    title = derive_idea_title(
        "A 5 nm Boron Nitride (BN) coating extends cycle life by >10% primarily "
        "by acting as a thermal heat spreader"
    )
    assert title == "A 5 nm Boron Nitride (BN) Coating"


def test_a_transitive_verb_whose_object_survived_the_cut_is_kept():
    title = derive_idea_title(
        "A 15 nm hybrid organic-inorganic coating extends cycle life by >10% at "
        "low discharge rates"
    )
    assert title == "A 15 nm Hybrid Organic-inorganic Coating Extends Cycle Life"


def test_a_participle_that_still_has_its_subject_is_kept():
    """The rule is about a dangling cut, not about the word ending in -ed."""
    title = derive_idea_title(
        "Cells coated with alumina retain capacity longer than uncoated "
        "controls do over five hundred cycles"
    )
    assert title == "Cells Coated with Alumina Retain Capacity Longer"


def test_titles_are_made_unique_without_falling_back_to_ids():
    claims = ["A coating improves cycle life"] * 3
    titles = unique_titles(claims)
    assert len(set(titles)) == 3
    assert all(title.startswith("A Coating Improves Cycle Life") for title in titles)


# The eight claims of a live run, which all answer the one goal and so all open on the
# same nine words, beside the eight names the generators wrote for them in the field
# the report used to discard.
_LIVE_CLAIMS = [
    (
        "A 2 nm ALD Al2O3 coating on NMC811 improves cycle life by pinning surface "
        "oxygen atoms via strong Al-O-TM bonds.",
        "Al2O3-Pinned Oxygen Lattice for Suppression of NMC811 Surface Reconstruction",
    ),
    (
        "A 2 nm ALD Al2O3 coating on NMC811 improves cycle life by reacting with "
        "trace HF to form an AlF3 passivating layer.",
        "Al2O3-Mediated HF Scavenging and AlF3 Interphase Formation",
    ),
    (
        "A 2 nm ALD Al2O3 coating acts as a wide-bandgap dielectric barrier, "
        "analogous to CMOS gate oxides, suppressing parasitic electron transfer.",
        "Wide-Bandgap Dielectric Passivation of NMC811 via ALD Al2O3 to Suppress "
        "Parasitic Electron Transfer",
    ),
]


def test_an_idea_is_named_by_the_generator_where_the_generator_named_one():
    """Every claim in a run answers the same goal, so every claim opens on the same
    words: a nine-word cut headed five of eight live sections "A 2 nm ALD Al2O3
    Coating", one of them "(Variant 2)". Each of those records already carried a
    name for itself in a field the report never read."""
    claims = [claim for claim, _ in _LIVE_CLAIMS]
    titles = unique_titles(claims, [name for _, name in _LIVE_CLAIMS])
    assert titles == [
        "Al2O3-Pinned Oxygen Lattice for Suppression of NMC811 Surface Reconstruction",
        "Al2O3-Mediated HF Scavenging and AlF3 Interphase Formation",
        "Wide-Bandgap Dielectric Passivation of NMC811 via ALD Al2O3 to Suppress "
        "Parasitic Electron Transfer",
    ]
    # What the same three claims are worth on their own, which is why the field is
    # read: three headings that open on the same six words and are told apart, if at
    # all, by whatever the ninth word happened to be.
    assert all(
        title.startswith("A 2 nm ALD Al2O3 Coating") for title in unique_titles(claims)
    )


def test_a_title_field_the_generator_left_empty_falls_back_to_the_claim():
    claims = ["A 5 nm ZrO2 coating suppresses transition metal dissolution"]
    assert unique_titles(claims, [""]) == unique_titles(claims)
    assert unique_titles(claims, ["Idea"]) == unique_titles(claims)


def test_a_title_field_holding_the_claim_again_is_cut_like_a_claim():
    """A generator that answers the title field with its own claim has named nothing,
    so the sentence still has to be turned into a heading rather than printed whole."""
    claim = (
        "Investigate whether a 5 nm ZrO2 coating suppresses transition metal "
        "dissolution in NMC811 because the fluoride scavenging pathway is blocked"
    )
    assert unique_titles([claim], [claim]) == unique_titles([claim])
    assert (
        unique_titles([claim])[0]
        == "5 nm ZrO2 Coating Suppresses Transition Metal Dissolution"
    )


def test_a_numbered_title_loses_the_number_the_heading_already_carries():
    assert (
        unique_titles(["An unrelated claim"], ["Hypothesis 3: Oxygen Lattice Pinning"])[
            0
        ]
        == "Oxygen Lattice Pinning"
    )


def test_two_ideas_of_one_name_are_told_apart_by_what_they_claim():
    """A number in a heading distinguishes nothing -- the reader is told two ideas
    share a name and never what either one says. The claims differ somewhere."""
    claims = [
        "A 2 nm Al2O3 coating improves cycle life by scavenging trace HF",
        "A 2 nm Al2O3 coating improves cycle life by pinning surface oxygen",
    ]
    titles = unique_titles(claims, ["Alumina Surface Chemistry"] * 2)
    assert len(set(titles)) == 2
    assert not any("Variant" in title for title in titles)
    assert titles[0].endswith("by Scavenging Trace")
    assert titles[1].endswith("by Pinning Surface")


def test_every_generator_is_told_what_the_title_it_writes_is_for():
    """The report prints that field as the section heading, so the brief that asks
    for it has to say what it is for -- otherwise a generator writes the goal back."""
    from coscientist.agents import STRUCTURED_OUTPUT_INSTRUCTIONS

    generating = [
        role for role in STRUCTURED_OUTPUT_INSTRUCTIONS if role.startswith("generation")
    ]
    assert len(generating) == 5
    for role in generating:
        instruction = STRUCTURED_OUTPUT_INSTRUCTIONS[role]
        assert "the heading the report prints over this idea" in instruction
        assert "not the goal restated" in instruction


def test_an_idea_named_on_its_own_is_named_the_way_the_population_names_it(
    rich_session: Session,
):
    """A withdrawn idea is named outside the one pass that titles the population, and
    a reader following a withdrawal back to the idea it withdrew needs one name."""
    from coscientist.models import Candidate

    record = load_record(rich_session)
    candidate = record.candidates[0]
    assert idea_title(candidate) == record.titles[candidate.id]
    named = Candidate(
        id="candidate_named",
        title="Fluoride Scavenging by a Sacrificial ZrO2 Overlayer",
        claim="A 5 nm ZrO2 coating suppresses transition metal dissolution",
        rationale="r",
        mechanism_model="m",
        validation_protocol="p",
        falsifier="f",
    )
    assert idea_title(named) == "Fluoride Scavenging by a Sacrificial ZrO2 Overlayer"


# ---------------------------------------------------------------- per-idea layout


def test_each_deep_dive_carries_rank_elo_and_a_category_path(body: str):
    deep_dives = body[body.rindex("\n# Top ideas in detail\n") :]
    ranks = re.findall(
        r"^Rank: (\d+)(?:, shared on Elo with .+)?$", deep_dives, re.MULTILINE
    )
    elos = re.findall(r"^Elo: (\d+)$", deep_dives, re.MULTILINE)
    categories = re.findall(r"^Category: (.+)$", deep_dives, re.MULTILINE)
    # An idea is numbered by its place in the standings, except that ideas which
    # finished level share the position of the first of them: 1, 2, 2, 4, not 1, 2, 3, 4.
    positions = [int(item) for item in ranks]
    assert positions[:1] == [1]
    for index, position in enumerate(positions[1:], start=2):
        assert position in (index, positions[index - 2]), (
            f"position {position} at place {index} is neither its own nor the one "
            "above it"
        )
    assert len(elos) == len(ranks)
    assert len(categories) == len(ranks)
    for category in categories:
        # Widest first, and no level repeated: the middle level is the cluster that
        # claimed the idea, so an idea no cluster claims has two levels rather than a
        # third padded out of the same fact the posture beside it is derived from.
        levels = category.split(" > ")
        assert levels[0] == "Experimental", f"category is not widest-first: {category}"
        assert 2 <= len(levels) <= 3, f"category is not a taxonomy path: {category}"
        assert len(set(levels)) == len(levels), f"category repeats a level: {category}"


def test_the_printed_elo_of_an_idea_is_where_its_own_match_table_ends(body: str):
    """Three separate roundings of one rating disagreed on the page by a point.

    The headline rounded the stored figure, each row rounded its own opening
    endpoint, and the row arithmetic added a rounded swing to it -- so a table could
    close at 1289 under a heading that said 1290, and one row could open a point
    below where the row above it closed. The printed arithmetic is the authority.
    """
    deep_dives = body[body.rindex("\n# Top ideas in detail\n") :]
    sections = deep_dives.split("\n## ")[1:]
    assert sections
    checked = 0
    for section in sections:
        headline = re.search(r"^Elo: (\d+)$", section, re.MULTILINE)
        rows = re.findall(
            r"^\| \d+ \| .+ \| \w+ \| (-?\d+) \| (-?\d+) \| .+ \|$",
            section,
            re.MULTILINE,
        )
        if headline is None or not rows:
            continue
        checked += 1
        for above, below in pairwise(rows):
            assert above[1] == below[0], "a row opens where the one above did not close"
        assert rows[-1][1] == headline.group(1), (
            f"the headline Elo {headline.group(1)} is not where the table ends"
        )
    assert checked, "no idea in this report played a ranked match"


def test_no_two_numbered_sections_carry_headings_a_reader_cannot_tell_apart(body: str):
    """ "5. Comparison of Candidate Ideas" and "6. Comparative Analysis of Candidate
    Ideas" sat on facing pages over a tournament and a review-score breakdown. The
    reference reports keep the pair distinct; the clone had taken the long form of
    both, so a reader looking for one of them had no way to know which they were in."""
    import difflib

    headings = re.findall(r"^#### \d+\. (.+)$", body, re.MULTILINE)
    assert len(headings) == 9
    for earlier, later in pairwise(headings):
        overlap = difflib.SequenceMatcher(None, earlier, later).ratio()
        assert overlap < 0.7, (
            f"{earlier!r} and {later!r} read as the same heading at {overlap:.0%}"
        )


def test_each_deep_dive_carries_the_eight_summary_subsections(body: str):
    deep_dives = body[body.rindex("\n# Top ideas in detail\n") :]
    expected = [f"##### {n}. {title}" for n, title in enumerate(SUMMARY_SUBSECTIONS, 1)]
    blocks = deep_dives.split("\n### Reviews\n")[1:]
    assert blocks
    for block in blocks:
        summary = block.split("\n#### Correctness\n")[0]
        assert [line for line in summary.splitlines() if line.startswith("##### ")] == (
            expected
        )


def test_every_scored_review_closes_on_a_matched_answer_and_score(body: str):
    """The answer says what the number means; printing the number twice said nothing."""
    pairs = re.findall(r"^Answer: (.+)\n\nScore: (\d+)$", body, re.MULTILINE)
    assert pairs
    assert all(not answer[0].isdigit() for answer, _ in pairs)
    assert all(1 <= int(score) <= 5 for _, score in pairs)
    for answer, score in pairs:
        # A five is an advance and nothing else can reach it. The confidence clause is
        # read off the reviewer rather than off the score, so a five carries one too --
        # which is the point: it is the only thing separating two advances that were
        # not held with anything like the same conviction.
        assert answer.startswith("advance the idea as written") == (score == "5")


@dataclass
class _Verdict:
    recommendation: str
    confidence: float
    fatal_flaws: list[str]


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        # An advance already scores five, so a score-derived clause had nowhere to go
        # and every advance in a live report read the same however it was held.
        (
            _Verdict("advance", 0.95, []),
            "advance the idea as written, at the reviewer's stated confidence of 0.95",
        ),
        (
            _Verdict("advance", 0.5, []),
            "advance the idea as written, at the reviewer's stated confidence of 0.50",
        ),
        (
            _Verdict("revise", 0.75, []),
            "revise it first, at the reviewer's stated confidence of 0.75",
        ),
        (
            _Verdict("advance", 0.9, ["unmitigated inhalation hazard"]),
            "advance the idea as written, at the reviewer's stated confidence of 0.90"
            ", with the score capped by a fatal flaw",
        ),
        # A reject is already at the floor, so the flaw caps nothing and the confidence
        # is the only thing left to say about it.
        (
            _Verdict("reject", 0.5, ["no mechanism"]),
            "reject it, at the reviewer's stated confidence of 0.50",
        ),
    ],
)
def test_the_answer_says_what_moved_the_score_off_the_recommendation(verdict, expected):
    """Two reviews can share a score for opposite reasons; the answer separates them."""
    assert _review_answer(verdict) == expected


def test_the_confidence_clause_states_the_figure_rather_than_a_band(body: str):
    """A band that is true of forty-six reviews out of forty-seven says nothing."""
    stated = re.findall(r"^Answer: .+ confidence of (\d\.\d\d)$", body, re.MULTILINE)
    assert stated
    assert len(set(stated)) > 1, (
        "the clause no longer distinguishes one review from another"
    )
    assert not re.search(r"^Answer: .+ (?:high|low) confidence$", body, re.MULTILINE)


def test_every_idea_carries_the_same_five_row_grid(body: str):
    tables = re.findall(
        r"\| Category \| Description \|\n\| --- \| --- \|\n((?:\|.*\|\n)+)", body
    )
    assert tables
    for table in tables:
        labels = [row.split("|")[1].strip().strip("*") for row in table.splitlines()]
        assert tuple(labels) == IDEA_TABLE_ROWS


def test_the_prose_beside_a_grid_does_not_reprint_the_grid(body: str):
    """A row of the grid and the paragraph above it are not two sources on one fact.

    Section 4 used to restate the rationale and the falsifier in full and then print
    both again, word for word, in the grid four lines below -- and open its prediction
    list on the sentence the grid had just given as the discriminating prediction.
    """
    listing = body[body.index("\n#### 4. Candidate Ideas\n") : body.index("\n#### 5. ")]
    slots = re.split(r"^##### 4\.\d+ ", listing, flags=re.MULTILINE)[1:]
    assert slots
    for slot in slots:
        prose, _, grid = slot.partition("| Category | Description |")
        assert grid, "every idea in the listing carries a grid"
        for row in grid.splitlines():
            cells = row.split("|")
            if len(cells) < 4 or cells[1].strip().startswith("---"):
                continue
            stated = " ".join(cells[2].split()).rstrip(".")
            if stated and stated != "None recorded":
                assert stated not in " ".join(prose.split()), (
                    f"the prose reprints the grid's {cells[1].strip().strip('*')} row"
                )


def test_each_deep_dive_ends_on_a_tournament_match_summary(body: str):
    deep_dives = body[body.rindex("\n# Top ideas in detail\n") :]
    blocks = deep_dives.split("\n### Tournament\n")[1:]
    assert blocks
    for block in blocks:
        assert block.startswith("\n#### Match summary\n")
        for label in (
            "Total matches",
            "Matches won",
            "Matches lost",
            "Matches tied",
            "Win rate",
        ):
            assert f"- {label}:" in block


def test_a_judged_debate_is_rendered_as_prose_turns(body: str):
    assert "### Debate against " in body
    assert "llm_debate" not in body, "the judge key leaked instead of being named"


def _undebated_match(round_number: int, rationale: str) -> SimpleNamespace:
    return SimpleNamespace(
        round_number=round_number,
        opponent_title=f"Opponent {round_number}",
        outcome="win",
        shown_before=1200,
        shown_after=1216,
        judge="deterministic",
        debate_turns=[],
        unreadable_turns=0,
        rationale=rationale,
    )


def _match_block(*matches: SimpleNamespace) -> str:
    brief = SimpleNamespace(
        matches=list(matches), wins=len(matches), losses=0, ties=0, win_rate=100
    )
    return "\n".join(_match_summary(brief))


def test_one_reason_reused_across_matches_is_reported_once_as_one_reason():
    """A judge that writes a line and reuses it verbatim has given a reason for the
    set, not for each match. Printed per round it read as four separate findings, and
    over seven ideas the report carried twenty-four copies of one sentence."""
    reason = "Compared evidence status, validity, novelty, feasibility, and impact."
    block = _match_block(*[_undebated_match(index, reason) for index in range(1, 5)])

    assert block.count(reason.rstrip(".")) == 1
    assert "The same reason is recorded for every one of them" in block
    assert "- **Round" not in block


def test_reasons_that_differ_are_still_printed_against_the_match_they_explain():
    block = _match_block(
        _undebated_match(1, "The first idea isolates the mechanism."),
        _undebated_match(2, "The second idea's control is confounded."),
    )

    assert block.count("- **Round") == 2
    assert "The same reason is recorded" not in block


def test_a_reason_list_shorter_than_the_match_count_says_so():
    """ "the judge's stated reason for each is below" over a list two bullets short of
    the total above sends the reader hunting for bullets nobody wrote."""
    block = _match_block(
        _undebated_match(1, "The first idea isolates the mechanism."),
        _undebated_match(2, ""),
    )

    assert block.count("- **Round") == 1
    assert "the one that recorded a reason is below" in block
    assert "the rest recorded no reason either" in block


def _brief_with(*matches: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        matches=list(matches), wins=len(matches), losses=0, ties=0, win_rate=100
    )


def _debated_match(round_number: int, opponent: str, turn: str) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **vars(_undebated_match(round_number, "The mechanism was stronger.")),
            "opponent_title": opponent,
            "debate_turns": [turn],
            "judge": "llm_debate",
            "confidence": 0.7,
        }
    )


def test_an_exchange_is_reproduced_under_one_of_the_two_ideas_that_played_it():
    """A match has two sides and each idea has a chapter, so every transcript was
    printed twice -- about six thousand words of a live report, and nothing on the
    second copy said it was the same exchange the reader had already met."""
    turn = (
        "Expert A: The coating thickness argument does not survive the transport data."
    )
    first = SimpleNamespace(
        **{
            **vars(_brief_with(_debated_match(2, "Second idea", turn))),
            "title": "First idea",
        }
    )
    second = SimpleNamespace(
        **{
            **vars(_brief_with(_debated_match(2, "First idea", turn))),
            "title": "Second idea",
        }
    )

    transcribed: set = set()
    blocks = [
        "\n".join(_match_summary(brief, frozenset(), transcribed))
        for brief in (first, second)
    ]

    assert blocks[0].count("does not survive the transport data") == 1
    assert "does not survive the transport data" not in blocks[1]
    assert "#### Debate against First idea" in blocks[1], (
        "the second chapter still has to say the match happened"
    )
    assert "reproduced there rather than in both chapters" in blocks[1]
    assert "Judge" in blocks[1], "each side keeps its own result row"


def test_the_same_pair_meeting_twice_keeps_both_exchanges():
    """The Swiss rounds and the top round robin can put two ideas together twice, and
    the second meeting is a different argument."""
    first = SimpleNamespace(
        **{
            **vars(
                _brief_with(
                    _debated_match(1, "Second idea", "Expert A: Round one point."),
                    _debated_match(4, "Second idea", "Expert A: Round four point."),
                )
            ),
            "title": "First idea",
        }
    )

    block = "\n".join(_match_summary(first, frozenset(), set()))

    assert "Round one point" in block
    assert "Round four point" in block


def test_one_judge_over_the_whole_tournament_is_named_once_above_the_ideas():
    """Who decided the matches nobody argued is the tournament's fact, not the fact of
    each idea whose table they appear under. On the adjudication run that sentence, and
    the single reason under it, were printed under all seven ideas."""
    reason = "Compared evidence status, validity, novelty, feasibility, and impact."
    briefs = [
        _brief_with(*[_undebated_match(index, reason) for index in range(1, 3)])
        for _ in range(3)
    ]

    notes, hoisted = shared_match_notes(briefs)
    blocks = [_match_summary(brief, hoisted) for brief in briefs]

    assert hoisted == frozenset({"judges", "reason", "tail"})
    assert "\n".join(notes).count(reason.rstrip(".")) == 1
    for block in blocks:
        assert "None of these matches carries a debate transcript." in block
        assert reason.rstrip(".") not in "\n".join(block)
        assert "decided by" not in "\n".join(block)


def test_reasons_that_differ_still_lose_the_sentence_saying_where_they_are():
    """Where every unargued match recorded a reason, "the judge's stated reason for
    each is below and the exchange behind it is not" is a fact about the layout and was
    printed under all eight ideas. The reasons themselves still differ and stay put."""
    briefs = [
        _brief_with(_undebated_match(1, f"Idea {index} isolates the mechanism."))
        for index in range(1, 4)
    ]

    notes, hoisted = shared_match_notes(briefs)
    block = "\n".join(_match_summary(briefs[0], hoisted))

    assert "tail" in hoisted and "reason" not in hoisted
    assert "and the reason the judge recorded" in "\n".join(notes)
    assert "the judge's stated reason for each is below" not in block
    assert "Idea 1 isolates the mechanism" in block


def test_a_match_with_no_recorded_reason_is_not_reported_as_having_one():
    """The hoisted note asserted a stated reason for every unargued match; a judge that
    recorded none left the report claiming a reason the reader cannot find."""
    briefs = [
        _brief_with(
            _undebated_match(1, "Compared the scores."), _undebated_match(2, "")
        )
        for _ in range(2)
    ]

    notes, hoisted = shared_match_notes(briefs)

    assert "tail" not in hoisted
    assert "a stated reason where the judge recorded one" in "\n".join(notes)
    assert "the rest recorded no reason either" in "\n".join(
        _match_summary(briefs[0], hoisted)
    )


def test_ideas_judged_differently_keep_the_judge_beside_their_own_matches():
    """One sentence naming both judges would attribute both to both ideas."""
    arithmetic = _undebated_match(1, "Compared the scores.")
    single_pass = SimpleNamespace(**{**vars(arithmetic), "judge": "llm_single_pass"})

    notes, hoisted = shared_match_notes(
        [_brief_with(arithmetic), _brief_with(single_pass)]
    )

    assert (notes, hoisted) == ([], frozenset())
    assert "decided by an arithmetic score comparison" in "\n".join(
        _match_summary(_brief_with(arithmetic), hoisted)
    )


# ------------------------------------------------------------ integrity surfacing


def test_a_deterministic_fallback_is_warned_about_once_in_the_warnings_chapter(
    body: str, report: str
):
    """The notice used to be printed under every overview section the stage fed and
    again under every idea, which on a run where one stage fell back put the same
    paragraph on six pages. It is a fact about the run, so it is stated once in the
    warnings chapter, and the body carries the count."""
    notice = "a fixed template, not the specialist's own reasoning"
    appendix = report[report.index(_APPENDIX) :]
    assert notice in appendix
    assert notice not in body
    assert "## Stages that produced a template rather than reasoning" in appendix
    # The table of what each stage produced stays where it was: it is a record of
    # the run, not a warning about it, and the warning points at it.
    assert "What each stage produced" in appendix
    assert "limitations apply to this report as a whole" in body


def test_several_stages_falling_back_at_once_are_not_warned_about_as_one():
    """Only the verb agreed with the count, so three stages shared one specialist.

    "the clustering by mechanism, evolution of the shortlist, and meta-review in this
    section are a fixed template, not the specialist's own reasoning" reported one
    template and one malformed answer behind three named stages.
    """
    from coscientist.advisories import _templated_stage_advisory
    from coscientist.narrative import ProvenanceNote, ResearchRecord

    def _note(stage: str, agent: str) -> ProvenanceNote:
        return ProvenanceNote(
            stage=stage,
            agent=agent,
            schema_name="",
            source="deterministic_fallback",
            repairs=[],
            error="",
        )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.provenance = [
        _note("proximity", "proximity_agent"),
        _note("evolve", "evolution_agent"),
        _note("meta_review", "meta_review_agent"),
    ]
    several = _templated_stage_advisory(record)[0].body
    assert "are fixed templates, not the specialists' own reasoning" in several
    assert (
        "Their answers came back incomplete or malformed and were replaced" in several
    )
    assert "re-run the stages before relying" in several

    record.provenance = record.provenance[:1]
    alone = _templated_stage_advisory(record)[0].body
    assert "is a fixed template, not the specialist's own reasoning" in alone
    assert "The specialist's answer came back" in alone
    assert "re-run the stage before relying" in alone


def test_a_repaired_payload_is_recorded_in_the_appendix(report: str):
    appendix = report[report.index(_APPENDIX) :]
    assert "repaired" in appendix
    assert "coerced confidence to a float" in appendix


def _run_block(report: str) -> list[str]:
    """The bullets under the appendix's Run heading, which are the run's facts."""
    block = report[report.index("\n## Run\n") :]
    return [line for line in block.splitlines()[1:] if line.startswith("- ")]


def _stage_table(report: str) -> list[str]:
    """The rows of the appendix table that says what each stage produced."""
    block = report[report.index("\n## What each stage produced\n") :]
    return [line for line in block.splitlines()[1:] if line.startswith("|")]


def test_a_specialist_is_named_in_the_appendix_as_the_report_names_it_elsewhere(
    rich_session: Session,
):
    """The table took the underscores out of the agent id instead of using the names
    every other part of the report uses, so a reader matching an inline warning about
    the "evidence and correctness review" to its row had to know that the run files
    that reviewer under "reflection"."""
    for artifact in rich_session.artifacts:
        if artifact.schema_name == "ReviewSet":
            artifact.agent = "reflection"
        if artifact.schema_name == "TournamentState":
            artifact.agent = "ranking"

    rows = " ".join(_stage_table(compile_dossier(rich_session)))

    assert "| evidence and correctness review |" in rows
    assert "| tournament ranking |" in rows
    assert "| reflection |" not in rows
    assert "| ranking |" not in rows


def test_a_specialist_that_produced_nothing_is_not_credited_with_the_stage(
    rich_session: Session,
):
    """The evidence row credited the Deep Research discovery specialist on a run whose
    Literature discovery section, two paragraphs above it, says that agent never ran."""
    manifest = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "DiscoveryManifest"
    )
    manifest.agent = "deep_research_discovery"
    manifest.payload["convergence_reason"] = "search_grounded_fallback"

    report = compile_dossier(rich_session)
    row = next(
        line for line in _stage_table(report) if "literature discovery manifest" in line
    )

    assert "The Deep Research agent did not run" in report
    assert "did not run" in row, (
        "the table credits a specialist the prose says never ran"
    )
    assert "see Literature discovery above" in row, "the row must point, not re-explain"


def _with_a_backfilled_review(session: Session) -> str:
    """One review of the top idea replaced by the placeholder the run backfills.

    That is the live shape: a reviewer answers for most of the ideas, the rest are
    filled in from the fixed template, and the stage is still recorded as the
    specialist's own because most of it was.
    """
    artifact = next(
        item for item in session.artifacts if item.schema_name == "ReviewSet"
    )
    artifact.payload["reviews"][0]["stood_in"] = True
    return artifact.payload["reviews"][0]["candidate_id"]


def test_a_review_no_reviewer_wrote_is_not_printed_as_that_reviewers_judgement(
    rich_session: Session,
):
    """The rank-1 idea of a live run carried a feasibility review that was parity's
    fixed template verbatim -- scored, counted in the ranking, and printed under the
    idea in the same prose as the three a reviewer had written."""
    _with_a_backfilled_review(rich_session)

    report = compile_dossier(rich_session)

    assert "review of this idea was written" in report
    assert "the verdict and score below are the placeholder's" in report


def test_a_backfilled_review_is_disclosed_where_the_run_says_nothing_was_substituted(
    rich_session: Session,
):
    """Provenance said "no stage falling back to a fixed template" over a report that
    printed one, because the substitution was inside a stage rather than of it."""
    _with_a_backfilled_review(rich_session)

    report = compile_dossier(rich_session)

    assert "no stage falling back to a fixed template" not in report
    assert "The review stage's answer was accepted with one review missing" in report
    assert "Reviews that no reviewer wrote" in report


def test_a_run_whose_reviewers_answered_for_every_idea_carries_no_such_warning(
    rich_session: Session,
):
    report = compile_dossier(rich_session)

    assert "Reviews that no reviewer wrote" not in report
    assert "review of this idea was written" not in report
    assert "is a placeholder" not in report


def test_a_criterions_mean_and_range_say_when_a_placeholder_is_inside_them(
    rich_session: Session,
):
    """The one place the report gives a spread per criterion, and on a live run the
    top of the feasibility range was the placeholder's score rather than anybody's
    judgement."""
    from coscientist.narrative import _review_summary

    _with_a_backfilled_review(rich_session)
    briefs = build_idea_briefs(load_record(rich_session))
    stood = next(
        review for brief in briefs for review in brief.reviews if review.stood_in
    )

    lines = _review_summary(briefs)
    line = next(item for item in lines if item.startswith(f"{stood.section}:"))

    assert "One of those is a placeholder" in line
    # Whichever figures this criterion's line prints: the reviewers of this fixture
    # agreed, so it gives a single score rather than a mean and a range.
    assert "counted in that figure as though it were one" in line
    # Only the criterion the reviewer skipped. The other four were answered in full.
    assert sum("placeholder" in item for item in lines) == 1


def _discovery_line(session: Session) -> str:
    """The sentence that says how the Deep Research stage went."""
    block = compile_dossier(session)
    heading = "\n## Literature discovery\n"
    block = block[block.index(heading) + len(heading) :]
    return next(line for line in block.splitlines() if line.strip())


def test_the_discovery_sentence_counts_its_passes_in_one_notation(
    rich_session: Session,
):
    """The two counts in this sentence came from two helpers, one printing a digit and
    one printing a word, so a run of a single pass read "1 pass, of which one
    completed" -- one quantity written twice, as though it were two."""
    manifest = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "DiscoveryManifest"
    )
    manifest.payload["runs"] = [
        {"pass_number": 1, "status": "completed"},
    ]

    line = _discovery_line(rich_session)

    assert line.startswith("Deep Research ran one pass, at an estimated cost of ")
    assert "1 pass" not in line
    assert "of which" not in line, "nothing to contrast when every pass completed"

    manifest.payload["runs"] = [
        {"pass_number": 1, "status": "completed"},
        {"pass_number": 2, "status": "failed", "error": "The provider timed out."},
    ]

    line = _discovery_line(rich_session)

    assert line.startswith("Deep Research ran two passes, of which one completed, ")


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        # The one a live appendix printed raw: "It stopped because the run recorded
        # gap_directed_search." -- an identifier out of the orchestrator, in the one
        # sentence a reader consults to judge how much of the field was searched.
        ("gap_directed_search", "the last passes were aimed at the gaps the fan-out"),
        ("deep_research_start_failed", "no pass of the search could be started"),
        (
            "fan_out_truncated_by_budget:negative_null,corrections_retractions",
            "the pass budget was reached before the fan-out was complete, so negative "
            "or null results and corrections or retractions affecting the sources "
            "used were never searched",
        ),
    ],
)
def test_a_stop_reason_is_written_in_words_and_not_as_its_token(
    rich_session: Session, reason: str, expected: str
):
    manifest = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "DiscoveryManifest"
    )
    manifest.payload["convergence_reason"] = reason

    line = _discovery_line(rich_session)

    assert expected in line
    assert reason not in line
    assert "the run recorded" not in line


def test_the_discovery_appendix_of_a_fork_says_the_search_was_not_its_own(
    rich_session: Session,
):
    """The Knowledge Summary opens by saying a forked run did not search, but that
    sentence guards one heading and this appendix is thirty pages below it. Standing
    alone, "Deep Research ran seven passes, at an estimated cost of $21.00" read as a
    plain fact about a run that ran none of them and spent nothing -- printed in the
    one section a reader goes to for exactly that number."""
    rich_session.seeded_evidence_from = "session_earlier"

    assert _discovery_line(rich_session).startswith(
        "The search below is not this run's."
    )
    assert "session_earlier" in _discovery_line(rich_session)
    # And the run's own provenance block says it too, because that is where a
    # reader checks what a run did -- and two lines around it are about work this
    # one did not do: the stage count includes the evidence stage it started past,
    # and the models named include the Deep Research model it never called.
    block = compile_dossier(rich_session).split("\n## Run\n", 1)[1]
    forked = next(
        line for line in block.splitlines() if line.startswith("- Evidence forked from")
    )
    assert "session_earlier" in forked
    assert "did not search the literature" in forked


def test_a_run_that_did_its_own_searching_carries_no_fork_note(rich_session: Session):
    report = compile_dossier(rich_session)
    assert "The search below is not this run's" not in report
    assert "Evidence forked from" not in report


def test_a_fork_does_not_claim_the_corpus_as_literature_it_gathered(
    rich_session: Session,
):
    """The Knowledge Base of a fork opens "This run did not search the literature",
    and two chapters above it the overview introduced the same corpus as "The
    literature this run gathered" and the novelty chapter as "the literature this run
    retrieved". The report claimed the search and disclaimed it, and a reader who met
    the claims first carried the wrong provenance into every novelty judgement."""
    rich_session.seeded_evidence_from = "session_earlier"

    report = compile_dossier(rich_session)

    assert "The literature this run gathered" not in report
    assert "the literature this run retrieved" not in report
    assert "The literature carried into this run from session_earlier" in report
    assert "the literature carried into this run from session_earlier" in report
    # The sentence the two contradicted is still the one that says where it came from.
    assert "This run did not search the literature." in report


def test_a_source_the_search_never_returned_is_accounted_for_where_it_is_counted(
    rich_session: Session,
):
    """The appendix said "returned 86 source leads" and the Knowledge Summary, twenty-
    two hundred lines above it, said "the literature search returned eighty-eight
    leads". Same search, two counts. The difference was two sources the evidence stage
    carried in, which the corpus admits so a claim resting on one can be numbered --
    recorded nowhere but in the code."""
    packet = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "EvidencePacket"
    )
    packet.payload["sources"].append(
        {
            "id": "source_carried",
            "url": "https://example.org/carried-by-the-evidence-stage",
            "title": "A Paper The Search Did Not Return",
            "verification_status": "verified",
        }
    )

    report = compile_dossier(rich_session)
    discovery = report.split("## Literature discovery", 1)[1].split("\n## ")[0]
    searched = len(
        next(
            artifact
            for artifact in rich_session.artifacts
            if artifact.schema_name == "DiscoveryManifest"
        ).payload["source_leads"]
    )

    from coscientist.narrative import _plural

    assert f"returned {_plural(searched, 'source lead')}" in discovery
    assert "one further source was carried in from the evidence stage" in discovery
    # Both counts through the same helper: "returned eight source leads" beside
    # "the corpus is 9 leads" is one quantity written two ways.
    assert f"the corpus is {_plural(searched + 1, 'lead')}" in discovery


def test_a_search_that_found_the_whole_corpus_is_not_told_it_found_less(
    rich_session: Session,
):
    """There is nothing to reconcile when nothing was carried in."""
    discovery = compile_dossier(rich_session).split("## Literature discovery", 1)[1]

    assert "carried in from the evidence stage" not in discovery.split("\n## ")[0]


def test_a_stage_repeated_with_the_same_answer_is_one_row_and_a_count(
    rich_session: Session,
):
    """Six verification batches recorded six notes, and every column of all six held
    the same value. The live table printed the identical row six times over, which a
    reader can only read as a repeat until they have counted them."""
    packet = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "EvidencePacket"
    )
    for _ in range(3):
        rich_session.artifacts.insert(
            rich_session.artifacts.index(packet) + 1, packet.model_copy(deep=True)
        )

    rows = [
        row
        for row in _stage_table(compile_dossier(rich_session))
        if "evidence packet" in row
    ]

    assert len(rows) == 1, "the same row must not be printed once per repeat"
    assert "evidence packet, four of them" in rows[0]


def test_a_merge_of_other_agents_answers_is_not_labelled_a_failed_stage(
    rich_session: Session,
):
    """The generation aggregator folds four generators' answers together and calls no
    model. Left at the artifact default it reached this column as "a fixed template
    (not a model)" -- the phrase reserved for a specialist that failed -- one line
    above a sentence saying no stage fell back to a template."""
    population = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "CandidatePopulation"
    )
    population.agent = "generation_aggregator"
    population.producer_model = "deterministic-offline"
    population.payload_source = "specialist"

    row = next(
        line
        for line in _stage_table(compile_dossier(rich_session))
        if "hypothesis population" in line
    )

    assert "a merge of the specialists' answers (no model call)" in row
    assert "a fixed template" not in row


def test_the_integrity_lead_in_names_only_the_cases_the_run_recorded(
    rich_session: Session,
):
    """It promised four kinds of qualification over a list that held two of them, and
    told the reader each line stated one of the four."""
    for artifact in rich_session.artifacts:
        if artifact.schema_name == "CandidatePopulation":
            for candidate in artifact.payload["candidates"]:
                candidate["evidence_ids"] = ["claim_missing"]

    report = compile_dossier(rich_session)
    block = report[report.index("\n## Evidence integrity\n") :]
    lead = next(line for line in block.splitlines()[2:] if line.strip())

    assert "its evidence is absent from this session." in lead
    assert "was retracted" not in lead
    assert "one of the four" not in lead


def test_a_series_in_the_run_facts_is_written_as_a_series(rich_session: Session):
    """Both joins read wrong. "Produced by" separated whole phrases with semicolons and
    sorted them by capitalisation, and "Judged by" ran two noun phrases together on a
    comma with no conjunction at all."""
    models = [
        "gemini-3.1-pro-preview",
        "google_search_grounding",
        "deterministic-offline",
    ]
    for index, artifact in enumerate(rich_session.artifacts):
        artifact.producer_model = models[index % len(models)]

    facts = _run_block(compile_dossier(rich_session))
    produced = next(line for line in facts if line.startswith("- Produced by:"))
    judged = next(line for line in facts if line.startswith("- Judged by:"))

    assert produced == (
        "- Produced by: a fixed template (not a model), gemini-3.1-pro-preview, and "
        "Google Search grounding (model not recorded)"
    )
    assert judged == (
        "- Judged by: a multi-turn model debate and an arithmetic score comparison"
    )


def test_the_run_facts_say_what_approval_meant_on_this_run(rich_session: Session):
    """A reader who opens the report at the appendix — which is what auditing it looks
    like — could not tell whether a person had passed on any of this."""
    assert not any(
        "Approvals:" in line and "no stage acceptance" not in line
        for line in _run_block(compile_dossier(rich_session))
    ), (
        "a run with no recorded acceptance must not claim an approval regime it ran under"
    )

    for stage in ("scope", "evidence"):
        rich_session.decisions.append(
            HumanDecision(
                action=DecisionAction.ACCEPT,
                stage=stage,
                actor="auto_approval_policy",
                automatic=True,
                session_version=1,
            )
        )
    facts = _run_block(compile_dossier(rich_session))
    approvals = next(line for line in facts if line.startswith("- Approvals:"))

    assert "accepted automatically under the auto approval profile" in approvals
    assert "rather than a person's agreement" in approvals
    # Stated once and pointed at: the appendix must not restate the warning. It points
    # by heading, because the warnings were collected into an appendix of their own and
    # this line went on sending the reader to Research Goal, where they no longer are.
    assert AUTO_APPROVAL_WARNING in approvals
    assert f"## {AUTO_APPROVAL_WARNING}" in compile_dossier(rich_session)
    assert "Nobody read what those stages produced" not in approvals


def test_the_run_facts_separate_the_tournament_configured_from_the_one_played():
    """ "3 Swiss rounds then a top-4 round robin, 12 matches in all" was printed over a
    record whose last round holds three matches between four ideas — half a round
    robin. And "stopped without converging with a final round that moved a rating by 46
    points" attached the final round to "converging"."""
    from coscientist.dossier import _tournament_facts

    def tournament(pairs: list[tuple[int, str, str]], **kwargs):
        return SimpleNamespace(
            swiss_rounds=1,
            top_round_robin_size=4,
            converged=False,
            score_movement=0.01,
            comparisons=[
                SimpleNamespace(round_number=number, candidate_a_id=a, candidate_b_id=b)
                for number, a, b in pairs
            ],
            **kwargs,
        )

    short = _tournament_facts(
        tournament([(1, "a", "b"), (2, "a", "b"), (2, "c", "d"), (2, "a", "c")])
    )

    assert short[0] == (
        "Tournament protocol configured: 1 Swiss round then a top-4 round robin"
    )
    assert short[1] == (
        "Tournament as played: 2 rounds of 1 and 3 matches, 4 in all; its last round "
        "played 3 matches between 4 ideas, where a round robin over them is 6 matches"
    )
    assert short[2] == (
        "Tournament outcome: stopped without converging; its final round moved a "
        "rating by 12 points"
    )

    complete = _tournament_facts(
        tournament(
            [(1, "a", "b")]
            + [
                (2, left, right)
                for left, right in (
                    ("a", "b"),
                    ("a", "c"),
                    ("a", "d"),
                    ("b", "c"),
                    ("b", "d"),
                    ("c", "d"),
                )
            ]
        )
    )
    unplayed = _tournament_facts(tournament([(1, "a", "b")]))

    assert "round robin over them" not in complete[1]
    assert "the configured round robin was never played" in unplayed[1]


@pytest.mark.parametrize(
    ("support", "expected"),
    [
        ("grounded", "grounded."),
        ("partially_grounded", "partially grounded."),
        ("unverified", "unverified."),
        ("uncited", "uncited."),
        ("unknown", "not resolved."),
    ],
)
def test_a_benign_verdict_reads_as_a_statement_not_an_alarm(
    support: str, expected: str
):
    notice = support_notice(support, [])
    assert expected in notice
    assert not notice.startswith("Warning")


@pytest.mark.parametrize("support", ["unsupported", "discredited"])
def test_a_broken_grounding_verdict_is_unmissable(support: str):
    notice = support_notice(support, ["claim_001"])
    assert notice.startswith("Warning:")


def test_two_broken_citations_are_named_as_a_pair_and_not_as_a_short_list():
    """ "claim_1_2, and stmt_3_pass2" -- ids are noun phrases, and the clause joiner
    put a comma before the conjunction, which reads as a list with an item lost
    between the two that survived."""
    notice = support_notice("unsupported", ["claim_1_2", "stmt_3_pass2"])

    assert "— `claim_1_2` and `stmt_3_pass2`." in notice
    assert ", and" not in notice


def test_three_broken_citations_keep_the_series_comma():
    notice = support_notice("unsupported", ["claim_1_2", "claim_1_3", "stmt_3_pass2"])

    assert "— `claim_1_2`, `claim_1_3`, and `stmt_3_pass2`." in notice


def test_a_broken_grounding_with_no_id_recorded_still_reads_as_a_sentence():
    notice = support_notice("unsupported", [])

    assert "— an id it did not record. Nothing grounds" in notice


def test_a_grounding_verdict_several_ideas_share_is_explained_once():
    """Five ideas marked unverified printed the same three-line notice five times.

    The verdict is a field of the idea and stays under it; what the verdict means is
    the same sentence wherever it appears, so it belongs above the ideas that share it.
    A verdict only one idea carries is not hoisted -- that costs a page-turn and saves
    nothing -- and the two alarming verdicts are never hoisted at any count.
    """
    from coscientist.narrative import shared_support_notices

    text, hoisted = shared_support_notices(
        ["unverified", "unverified", "uncited", "grounded"]
    )
    assert hoisted == {"unverified"}
    # Of the four, so that the two ideas whose verdicts are not hoisted are visibly
    # unaccounted for rather than silently missing from the arithmetic.
    assert "Two of the four are marked unverified." in text
    assert "uncited" not in text
    assert "grounded" not in text

    quiet, none_hoisted = shared_support_notices(["unverified", "uncited"])
    assert (quiet, none_hoisted) == ("", set())

    # A reader must meet a broken grounding under the idea it belongs to, however
    # many ideas carry one.
    alarming, still_none = shared_support_notices(["unsupported", "unsupported"])
    assert (alarming, still_none) == ("", set())


def test_the_ideas_a_hoisted_verdict_leaves_out_are_not_read_as_the_clear_ones():
    """An alarming verdict is never hoisted, so on a live shortlist of eight the whole
    summary read "three of the eight are marked unverified" -- while four of the other
    five cited evidence that had been retracted. A reader who takes that sentence for
    the tally reads the four worst ideas in the report as the unremarkable rest. What
    the verdict means still belongs under the idea; that there is one does not."""
    from coscientist.narrative import shared_support_notices

    live = ["discredited"] * 4 + ["unverified"] * 3 + ["partially_grounded"]
    brief, _hoisted = shared_support_notices(live, detail=False)

    assert "three of the eight are marked unverified" in brief
    assert "The remaining five are not therefore clear" in brief
    # Still not hoisted: the warning itself is met under the idea it belongs to.
    assert "retracted" not in brief and "discredited" not in brief

    # Nothing to disclaim when every idea is accounted for by the counts above.
    covered, _ = shared_support_notices(["unverified"] * 3 + ["grounded"] * 2)
    assert "not therefore clear" not in covered

    one, _ = shared_support_notices(["unverified", "unverified", "discredited"])
    assert "The remaining idea is not therefore clear: it carries" in one


def test_what_a_shared_grounding_verdict_means_is_explained_in_one_place_only():
    """The verdicts are reported twice -- beside each idea in the ranked listing, and
    again above the deep dives. Explaining them in both put the same three lines in two
    sections; the second now carries the counts and points at the first."""
    from coscientist.narrative import shared_support_notices

    supports = ["unverified"] * 5 + ["uncited"] * 3
    explained, hoisted = shared_support_notices(supports)
    pointer, same = shared_support_notices(supports, detail=False)

    assert hoisted == same == {"unverified", "uncited"}
    assert "rests on retrieved text rather than on checked evidence" in explained
    # The counts stay in both: which ideas are affected is a fact of the run and the
    # reader of either section needs it. What it means is stated once.
    assert (
        "five of the eight are marked unverified and three others are marked uncited"
        in pointer
    )
    assert "rests on retrieved text" not in pointer
    assert "Candidate Ideas above" in pointer


def test_a_hoisted_grounding_verdict_is_still_named_beside_its_own_idea(
    rich_session: Session,
):
    """Hoisting the explanation must not take the verdict with it: a reader looking at
    one idea has to be able to see what its grounding was judged to be. Five ideas
    sharing a verdict printed its three-line body five times in the ranked listing."""
    from dataclasses import replace

    record = load_record(rich_session)
    briefs = [
        replace(brief, support="unverified", unresolved_evidence_ids=[])
        if not brief.support_is_alarming
        else brief
        for brief in build_idea_briefs(record)
    ]
    shared = [brief for brief in briefs if brief.support == "unverified"]
    assert len(shared) > 1, "the fixture no longer shares a verdict between ideas"

    draft = _section_four(record, briefs)
    body = "\n".join(
        [*draft.core, *[p for sub in draft.subsections for p in sub.paragraphs]]
    )
    explanation = "rests on retrieved text rather than on checked evidence"

    assert body.count(explanation) == 1
    assert body.count("Its grounding is marked unverified.") == len(shared)


def test_a_hoisted_verdict_under_a_deep_dive_says_where_it_is_explained(
    rich_session: Session,
):
    """The deep dive printed "Evidence support: unverified." and nothing else, a
    thousand lines below the paragraph that says what the word means -- and that page
    is where a reader is standing when they decide whether to act on the idea. The
    Executive Candidate Summary already says where its evidence column is explained."""
    from dataclasses import replace

    from coscientist.dossier import _idea_deep_dive

    record = load_record(rich_session)
    briefs = [
        replace(brief, support="unverified", unresolved_evidence_ids=[])
        if not brief.support_is_alarming
        else brief
        for brief in build_idea_briefs(record)
    ]
    shared = [brief for brief in briefs if brief.support == "unverified"]
    assert len(shared) > 1, "the fixture no longer shares a verdict between ideas"

    chapter = "\n".join(_idea_deep_dive(record, shared[0], grounding_hoisted=True))

    assert (
        "Evidence support: unverified — the verdict explained under Candidate Ideas "
        "above." in chapter
    )
    # Hoisted means hoisted: the explanation itself is still stated in one place.
    assert "rests on retrieved text rather than on checked evidence" not in chapter


def test_a_finding_more_than_one_idea_rests_on_is_reported_as_carrying_them_both(
    rich_session: Session,
):
    """Three ideas of a live run printed the same lone finding under Supporting
    Arguments. Each section was correct and none of them could say what the three
    together say: one finding is holding up three of the seven, and a fault in it
    takes all three. That is a fact about the field, so it goes above the ideas."""
    from coscientist.narrative import shared_grounding_reach

    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    stated = shared_grounding_reach(record, briefs)[0]

    assert stated.startswith("Of the six ideas below, more than one rests on the same ")
    assert "[2] by two of them" in stated
    assert "reaches further than the section reporting it" in stated
    # Once, above the ideas -- not under each idea that cites the finding.
    assert compile_dossier(rich_session).count(stated) == 1


def test_more_than_one_shared_finding_is_not_announced_as_the_same_one(
    rich_session: Session,
):
    """A live report opened this sentence "more than one rests on the same finding:"
    and then listed three of them -- [9] by five ideas, [11] by three, [12] by three.
    Read as an expansion of the singular, the colon promises one shared finding and
    the reader counts two more."""
    from coscientist.narrative import shared_grounding_reach

    record = load_record(rich_session)
    by_id = {candidate.id: candidate for candidate in record.candidates}
    # One more idea onto findings another idea already rests on, so two are shared.
    by_id["candidate_0006"].evidence_ids = ["claim_0", "claim_1", "claim_2"]
    briefs = build_idea_briefs(record)
    stated = shared_grounding_reach(record, briefs)[0]

    assert stated.startswith(
        "Of the six ideas below, more than one rests on each of two findings: "
    )
    assert "the same finding" not in stated


def test_a_field_whose_ideas_share_no_finding_says_nothing_about_sharing(
    rich_session: Session,
):
    """The overlap is only worth a paragraph where there is one."""
    from coscientist.narrative import shared_grounding_reach

    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    # Every idea on a finding nobody else cites.
    for index, candidate in enumerate(record.population.candidates):
        candidate.evidence_ids = [f"claim_{index}"]

    assert shared_grounding_reach(record, briefs) == []


def test_a_missing_evidence_id_is_never_presented_as_grounding(
    rich_session: Session, body: str
):
    briefs = build_idea_briefs(load_record(rich_session))
    broken = [brief for brief in briefs if brief.support_is_alarming]
    assert broken, "the fixture no longer exercises a broken grounding"
    for brief in broken:
        assert brief.support_notice in body
        # The idea is still named and still readable; it is the grounding that
        # is withdrawn, not the idea.
        assert brief.title in body


def test_an_id_invented_in_a_statement_is_counted_by_the_warning_that_names_them(
    rich_session: Session,
):
    """Three reviews of a live idea named three fabricated ids; the warning at the head
    of that idea named two, because the third was written into an evidence statement
    rather than into the field the resolver reads. The audit trail printed "no record of
    that id exists" against it, so the report held both counts and printed the low one
    where a reader would act on it."""
    from coscientist.narrative import _integrity_entries

    record = load_record(rich_session)
    broken = next(
        candidate
        for candidate in record.candidates
        if record.evidence_support[candidate.id].support == "unsupported"
    )
    resolver_saw = record.evidence_support[broken.id].unresolved
    assert resolver_saw, "the fixture no longer exercises an invented citation"
    record.cited_evidence[broken.id][0].append("stmt_4_pass1")

    brief = next(
        item for item in build_idea_briefs(record) if item.candidate_id == broken.id
    )
    assert "`stmt_4_pass1`" in brief.support_notice
    for named in resolver_saw:
        assert f"`{named}`" in brief.support_notice

    line = next(
        text
        for _, text in _integrity_entries(record)
        if record.title_for(broken.id) in text
    )
    assert "`stmt_4_pass1`" in line, (
        "the run-level list and the idea's own warning disagree on the count"
    )


def test_an_ordinary_statement_is_not_read_as_an_invented_identifier(
    rich_session: Session,
):
    """Only a statement that is nothing but an id is one. A sentence that happens to
    contain an underscored token -- a formula, a filename -- is a statement."""
    record = load_record(rich_session)
    broken = next(
        candidate
        for candidate in record.candidates
        if record.evidence_support[candidate.id].support == "unsupported"
    )
    before = len(record.evidence_support[broken.id].unresolved)
    record.cited_evidence[broken.id][0].append(
        "Coating Li_2CO_3 residue was measured at pass_2 of the deposition."
    )

    brief = next(
        item for item in build_idea_briefs(record) if item.candidate_id == broken.id
    )
    assert brief.unresolved_evidence_ids == list(
        record.evidence_support[broken.id].unresolved
    )
    assert len(brief.unresolved_evidence_ids) == before


def test_every_idea_states_a_support_verdict_somewhere_it_is_shown(
    rich_session: Session, body: str
):
    briefs = build_idea_briefs(load_record(rich_session))
    assert {brief.support for brief in briefs} >= {
        "grounded",
        "unsupported",
        "discredited",
        "uncited",
    }, "the fixture no longer covers the interesting verdicts"
    for brief in briefs:
        assert brief.support_notice in body


def test_an_id_quoted_inside_specialist_prose_is_replaced_by_what_it_names(
    rich_session: Session,
):
    """Reviewers write ids into their own sentences; the reader cannot resolve them."""
    reviews = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "ReviewSet"
    )
    reviews.payload["reviews"][0]["findings"] = [
        "The candidate relies entirely on claim_2, which is marked unverified",
        "It also leans on claim_absent, which nothing in the run defines",
    ]
    body = compile_dossier(rich_session).split(_APPENDIX)[0]
    _assert_no_record_ids(body)
    assert "the unverified claim drawn from" in body, (
        "a resolvable id lost its referent"
    )
    # Set as the identifier it is. Three unresolvable ids described in words read
    # "(a record this session does not hold, a record this session does not hold, a
    # record this session does not hold)", which names none of them.
    assert "`claim_absent`" in body


def test_an_id_capitalised_at_the_start_of_a_sentence_still_resolves(
    rich_session: Session,
):
    """The id regex is case-insensitive; the lookup behind it was not.

    A reviewer opening a sentence with an id capitalises it, and "Claim_1 and the
    source ... already explore dry-coating methods" then missed the ``claim_1`` the
    record holds. What printed in its place was the renderer's own miss message, as
    the subject of the reviewer's sentence.
    """
    reviews = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "ReviewSet"
    )
    reviews.payload["reviews"][0]["findings"] = [
        "Claim_2 is the only support offered for the central mechanism"
    ]
    body = compile_dossier(rich_session).split(_APPENDIX)[0]
    assert "`Claim_2`" not in body
    assert "The unverified claim drawn from" in body, (
        "the sentence lost its opening capital"
    )


def _findings(session: Session, *findings: str) -> str:
    reviews = next(
        artifact
        for artifact in session.artifacts
        if artifact.schema_name == "ReviewSet"
    )
    reviews.payload["reviews"][0]["findings"] = list(findings)
    return compile_dossier(session).split(_APPENDIX)[0]


def test_an_id_with_more_than_one_numbered_part_still_resolves(rich_session: Session):
    """A live transcript argued an idea up the ranking because it "relies on verified
    evidence (claim_11_1, source_11_2)" -- two records the run never retrieved, whose
    ids the pattern missed because it stopped at the first numbered part."""
    evidence = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "EvidencePacket"
    )
    evidence.payload["claims"].append(
        {
            "id": "claim_11_1",
            "claim": "A 2.5 nm coating held 82 per cent of capacity at 500 cycles.",
            "source_id": "source_0",
            "verification_status": "discovered_unverified",
            "confidence": 0.5,
            "relation": "supports",
        }
    )
    body = _findings(rich_session, "The case rests on claim_11_1 and nothing else")

    _assert_no_record_ids(body)
    assert "the unverified claim drawn from" in body


def test_a_verified_record_is_not_labelled_as_though_it_were_doubtful(
    rich_session: Session,
):
    body = _findings(rich_session, "The mechanism rests on claim_0")

    assert "the claim drawn from Thin-film passivation" in body
    assert "unverified claim drawn from Thin-film passivation" not in body


def test_two_claims_whose_sources_went_unnamed_are_still_told_apart(
    rich_session: Session,
):
    """A live review read "(the unretrieved claim drawn from Limitations of Ultrathin
    Al2O3 Coatings, the unverified cited claim, the unverified cited claim)" -- two
    different records inside one parenthesis under the same four words, naming neither
    which claims the reviewer meant nor what either of them held."""
    evidence = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "EvidencePacket"
    )
    for index, text in (
        (1, "Pore blockage alone accounts for the retention gain."),
        (2, "Interphase composition sets the rate of lithium transport."),
    ):
        evidence.payload["claims"][index]["source_id"] = ""
        evidence.payload["claims"][index]["claim"] = text
    body = _findings(rich_session, "The idea synthesizes claim_1 and claim_2")

    _assert_no_record_ids(body)
    assert "claim that pore blockage alone accounts for the retention gain" in body
    assert (
        "claim that interphase composition sets the rate of lithium transport" in body
    )
    assert "cited claim" not in body


def test_a_claim_and_its_own_source_cited_together_name_the_paper_once(
    rich_session: Session,
):
    """A specialist cites a claim and the source it was drawn from as a bracketed
    pair, and both ids resolve to the same document: "(the claim drawn from Hanyang
    team pinpoints 2.5nm minimum coating, the source Hanyang team pinpoints 2.5nm
    minimum coating)" ran twice in one paragraph of a live report, reading as two
    papers where the run holds one. The rule that folded them only knew the "and"
    form."""
    body = _findings(
        rich_session, "Literature shows a 43% gain (claim_1, source_1) at 500 cycles."
    )

    _assert_no_record_ids(body)
    finding = next(
        line for line in body.splitlines() if "Literature shows a 43% gain" in line
    )
    assert "claim drawn from Binder chemistry" in finding
    assert finding.count("Binder chemistry") == 1, (
        "the same paper is named twice inside one citation"
    )
    assert "and that source" in finding


def test_a_review_writing_about_the_identifiers_keeps_the_identifiers(
    rich_session: Session,
):
    """A live review read "cites invalid evidence IDs (claim_4, the unverified cited
    claim) which are not in the citable evidence list", and its next sentence listed
    the paper claim_4 names among the valid ones. Naming an id inside a sentence about
    ids makes the report accuse a correctly cited paper of being a fabricated
    identifier, and leaves the reader nothing to check against the evidence list."""
    body = _findings(
        rich_session,
        "The idea cites invalid evidence IDs (claim_1, claim_9) which are not in "
        "the citable evidence list.",
        "The valid ids are used correctly to establish the baseline.",
        "Its mechanism rests on claim_1, which is sound.",
    )

    assert "invalid evidence IDs (`claim_1`, `claim_9`)" in body
    # Everywhere else the id is still the paper it names.
    assert "Its mechanism rests on the claim drawn from Binder chemistry" in body


def test_a_claim_too_long_to_splice_says_only_what_kind_of_record_it_is(
    rich_session: Session,
):
    evidence = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "EvidencePacket"
    )
    evidence.payload["claims"][2]["source_id"] = ""
    body = _findings(rich_session, "The idea rests on claim_2")

    _assert_no_record_ids(body)
    assert "the unverified cited claim" in body


def test_a_retracted_record_says_so_where_the_reviewer_cites_it(
    rich_session: Session,
):
    body = _findings(rich_session, "The benchmark comes from source_3")

    assert "the retracted source" in body


def test_a_source_lead_cited_by_id_is_named_by_what_it_is(rich_session: Session):
    """ "(lead_0f651732f8364b01)" was printed inside an idea's own description."""
    record = load_record(rich_session)
    lead = record.discovery.source_leads[2]
    body = _findings(rich_session, f"The competing reading comes from {lead.id}")

    _assert_no_record_ids(body)
    assert "the unverified source Solid electrolyte interphase" in body


def test_a_short_discovery_finding_cited_by_id_is_read_out_where_it_is_cited(
    rich_session: Session,
):
    discovery = next(
        artifact
        for artifact in rich_session.artifacts
        if artifact.schema_name == "DiscoveryManifest"
    )
    statement = discovery.payload["narratives"][0]["statements"][0]
    statement["text"] = "Coated cells lose ten per cent less capacity by cycle 500."
    body = _findings(rich_session, f"This is answered by {statement['id']}")

    _assert_no_record_ids(body)
    assert "the finding that coated cells lose ten per cent less" in body


def test_a_finding_too_long_to_splice_is_named_rather_than_read_out(
    rich_session: Session,
):
    """A finding is a sentence, and a sentence spliced into the middle of a
    reviewer's own sentence is only readable while it is short.

    Where the run holds more than one such finding they cannot all be called the same
    thing, so each opens with its own first few words and is shown as cut. What must
    not happen either way is the whole finding arriving inside the reviewer's clause.
    """
    record = load_record(rich_session)
    statement = record.discovery.narratives[0].statements[0]
    body = _findings(rich_session, f"This is answered by {statement.id}")

    _assert_no_record_ids(body)
    named = "the finding that atomic layer deposition of alumina on silicon anodes …"
    # The ellipsis closes the sentence. This asserted a full stop after it, which is
    # how "conventional carbonate …." reached a live report.
    assert f"This is answered by {named}" in body
    assert f"{named}." not in body
    # The finding itself is printed in full in the Knowledge Base, which is where a
    # reader who wants the whole of it goes; what the review carries is a name.
    assert statement.text in body


def test_a_waived_evidence_gate_is_a_blocking_warning_the_body_counts(
    rich_session: Session,
):
    """Waived evidence is exploratory; the report must not let it read as verified.

    The paragraph itself now sits in the warnings chapter, so what the body owes the
    reader is the fact that a blocking warning exists and what it is called. A count
    with no name would be a footnote marker.
    """
    rich_session.exploratory_evidence_accepted = True
    report = compile_dossier(rich_session)
    body, appendix = report.split(_APPENDIX)[0], report[report.index(_APPENDIX) :]
    assert "The evidence gate for this run was waived" in appendix
    assert "exploratory leads rather than findings" in appendix
    assert "the evidence gate for this run was waived" not in body.lower()
    assert "a waived evidence gate" in body
    assert "should not proceed on the material" in body


def test_an_auto_approved_run_says_no_human_inspected_it(rich_session: Session):
    """Auto approval is a convenience, and the report has to keep saying so.

    Only where it happened, though. The sentence used to be unconditional, so a run
    a person had answered every gate on still disclaimed the automation it had not
    used, and the reader had one more standing caveat to learn to skip.
    """
    from coscientist.models import DecisionAction, HumanDecision

    rich_session.decisions.append(
        HumanDecision(
            action=DecisionAction.ACCEPT,
            stage="generate",
            actor="auto_policy",
            automatic=True,
            session_version=rich_session.version,
        )
    )
    report = compile_dossier(rich_session)
    body, appendix = report.split(_APPENDIX)[0], report[report.index(_APPENDIX) :]
    assert "Approval profile: auto" in body
    assert "## Stage gates approved without a human" in appendix
    assert "Auto approval is a workflow convenience" in appendix
    assert (
        "never constitutes scientific, safety, ethics, or institutional approval"
        in appendix
    )


def test_the_cover_names_every_scoring_reviewer_as_the_report_names_it(body: str):
    """The Attributes list says who scores each dimension, and it has to say it in the
    words the review chapters use. Hand-written, it invented two: a live cover read
    "Feasibility: scored one to five by the methods and feasibility review" and
    "Safety: scored one to five by the safety and governance review" over a report
    whose reviewers are the Methods and statistics reviewer and the Ethics, safety and
    governance reviewer, leaving a reader no way to tell whether that was one pass or
    two."""
    from coscientist.narrative import _REVIEWER_NAMES, CRITERION_SECTIONS
    from coscientist.parity import REVIEW_CRITERIA

    attributes = body.split("## Attributes", 1)[1].split("\n## ", 1)[0]
    for reviewer, (criterion, _label) in REVIEW_CRITERIA.items():
        line = f"{CRITERION_SECTIONS[criterion]}: scored one to five by "
        assert line in attributes, line
        assert f"{line}the {_REVIEWER_NAMES[reviewer].lower()}" in attributes


def test_a_real_workflow_run_produces_the_same_document_shape():
    """The fixture is a stand-in; the shape has to hold for the workflow's own output.

    A fixture can drift into being the only input the renderer is ever exercised on,
    at which point it tests the fixture. This runs the deterministic pipeline end to
    end and re-applies the two properties that matter most: the part order and the
    absence of artifact debris.
    """
    flow = CoScientistWorkflow(
        "Can a protective coating improve lithium-ion battery cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    flow.run_auto()
    live_body = flow.render_report().split(_APPENDIX)[0]
    test_the_nine_parts_appear_once_each_and_in_the_reference_order(live_body)
    test_the_narrative_has_exactly_nine_numbered_sections_in_order(live_body)
    test_every_hash_in_the_body_is_a_heading(live_body)
    # Fences are for diagrams and nothing else. A fence with any other tag, or
    # none, is a serialized artifact that escaped into the prose.
    fences = re.findall(r"^```(.*)$", live_body, flags=re.M)
    assert fences[0::2] == ["mermaid"] * (len(fences) // 2)
    assert fences[1::2] == [""] * (len(fences) // 2)
    diagrams = "\n".join(re.findall(r"^```mermaid$.*?^```$", live_body, re.M | re.S))
    prose = live_body.replace(diagrams, "") if diagrams else live_body
    assert "{" not in prose and "}" not in prose
    assert "http" not in prose
    assert not re.search(r"\bN/?A\b", live_body)
    assert not re.search(r"[0-9a-f]{12,}", live_body)
    _assert_no_record_ids(live_body)


def test_the_appendix_lists_every_idea_whose_grounding_does_not_hold(
    rich_session: Session, report: str
):
    briefs = build_idea_briefs(load_record(rich_session))
    appendix = report[report.index(_APPENDIX) :]
    assert "## Evidence integrity" in appendix
    for brief in briefs:
        if brief.support_is_alarming:
            assert brief.title in appendix


def test_the_appendix_says_when_it_is_every_idea_and_not_only_the_ones_listed(
    rich_session: Session, report: str
):
    """The Evidence integrity list groups by case rather than by idea, so no line in it
    counts the ideas it covers. On both live runs it covered every idea in the run,
    under a lead-in reading "the following ideas" -- and a reader had to tally its
    titles against the population to learn that nothing in the run was verified."""
    from coscientist.dossier import _provenance_appendix

    record = load_record(rich_session)
    assert not all(item.qualified for item in record.evidence_support.values()), (
        "the fixture must ground one idea, or there is no partial case to check"
    )
    assert "Each of the following ideas carries a qualification on its grounding:" in (
        report
    )

    record.evidence_support = {
        key: item for key, item in record.evidence_support.items() if item.qualified
    }
    every = "\n".join(_provenance_appendix(record))
    assert "Every idea in this run carries a qualification on its grounding:" in every
    assert "the following ideas" not in every


def test_the_stage_column_spells_a_stage_the_way_the_rest_of_the_report_does(
    rich_session: Session,
):
    """meta_review reached the column as "meta review", the id with its underscore
    swapped for a space, in a report that writes "meta-review" everywhere else --
    including in the Specialist cell of the same row."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import ProvenanceNote

    record = load_record(rich_session)
    record.provenance.append(
        ProvenanceNote(
            stage="meta_review",
            agent="meta_review",
            schema_name="MetaReview",
            source="specialist",
            repairs=[],
            error="",
            model="gemini-3.1-pro-preview",
        )
    )
    stages = [
        row.split("|")[1].strip()
        for row in _stage_table("\n".join(_provenance_appendix(record)))[2:]
    ]
    assert "meta-review" in stages
    assert "meta review" not in stages
    # The ids themselves reached the column too. A reader joining this table to the
    # chapter above it has no way to know that "reflect" is the review pass.
    assert not {"scope", "evidence", "generate", "reflect"} & set(stages)
    assert {"scoping the goal", "literature discovery", "idea generation"} <= set(
        stages
    )


def test_a_stage_run_by_one_specialist_of_its_own_name_says_so_once(
    rich_session: Session,
):
    """Naming the stage in words rather than by its id makes five of the nine rows
    print the one name twice: "| tournament ranking | tournament ranking |", and
    "| scoping the goal | goal scoping |" for the same name with the words swapped.
    A stage that did fan out still names each specialist that worked on it."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import ProvenanceNote

    def note(stage: str, agent: str, schema: str) -> ProvenanceNote:
        return ProvenanceNote(
            stage=stage,
            agent=agent,
            schema_name=schema,
            source="specialist",
            repairs=[],
            error="",
            model="gemini-3.1-pro-preview",
        )

    record = load_record(rich_session)
    record.provenance = [
        note("scope", "goal_manager", "ResearchPlan"),
        note("evidence", "evidence_discovery", "DiscoveryManifest"),
        note("evidence", "source_verification", "EvidencePacket"),
        note("rank", "ranking", "TournamentRecord"),
        note("evolve", "evolution", "RevisionCycle"),
        note("proximity", "proximity", "IdeaLandscape"),
        note("meta_review", "meta_reviewer", "MetaReview"),
    ]
    rows = [
        [cell.strip() for cell in row.strip("|").split("|")]
        for row in _stage_table("\n".join(_provenance_appendix(record)))[2:]
    ]
    named = {row[0]: row[1] for row in rows}
    assert named["scoping the goal"] == "the stage's only specialist"
    assert named["tournament ranking"] == "the stage's only specialist"
    assert named["evolution of the shortlist"] == "the stage's only specialist"
    assert named["clustering by mechanism"] == "the stage's only specialist"
    assert named["meta-review"] == "the stage's only specialist"
    # Two specialists worked the discovery stage, so each is worth naming.
    fanned = [row[1] for row in rows if row[0] == "literature discovery"]
    assert fanned == ["literature discovery", "source verification"]


def test_one_reason_behind_twenty_repairs_is_written_once(rich_session: Session):
    """A live report's repair paragraph read "Candidate.score_novelty 6 -> 3
    (answered on a 1-10 scale); Candidate.score_feasibility 9 -> 5 (answered on a
    1-10 scale); ..." for twenty fields, the identical parenthetical after every
    one. It is one finding about the stage, not twenty, and repeating it buries
    the field names the sentence exists to record."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import ProvenanceNote

    record = load_record(rich_session)
    record.provenance.append(
        ProvenanceNote(
            stage="reflect",
            agent="reflection",
            schema_name="ReviewSet",
            source="repaired",
            repairs=[
                "Candidate.score_novelty 6 -> 3 (answered on a 1-10 scale)",
                "Candidate.score_feasibility 9 -> 5 (answered on a 1-10 scale)",
                "ReviewSet.reviews: wrapped a scalar in a list",
                "Candidate.score_impact 80 -> 4 (answered on a 1-100 scale)",
            ],
            error="",
            model="gemini-3.1-pro-preview",
        )
    )
    appendix = "\n".join(_provenance_appendix(record))

    assert appendix.count("(answered on a 1-10 scale)") == 1
    assert (
        "candidate novelty score 6 → 3; candidate feasibility score 9 → 5 "
        "(answered on a 1-10 scale)" in appendix
    )
    # A different reason is a different finding and keeps its own parenthetical,
    # and a repair that carries no reason keeps its own detail.
    assert "Candidate impact score 80 → 4 (answered on a 1-100 scale)" in appendix
    assert "Review set reviews: wrapped a scalar in a list" in appendix
    # Nothing here is a path out of a Pydantic model.
    assert "Candidate.score" not in appendix
    assert "ReviewSet." not in appendix


def test_one_field_rescaled_for_every_idea_is_named_once_over_its_values(
    rich_session: Session,
):
    """A live Provenance chapter recorded a stage that rescaled five scores for
    each of four ideas as twenty entries, in which "Candidate.score_novelty"
    appeared four times carrying four different numbers with nothing saying the
    four were four ideas rather than four readings of one."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import ProvenanceNote

    record = load_record(rich_session)
    record.provenance.append(
        ProvenanceNote(
            stage="evolve",
            agent="evolution",
            schema_name="CandidateSet",
            source="repaired",
            repairs=[
                f"Candidate.{field} {raw} -> {raw // 2} (answered on a 1-10 scale)"
                for raw in (8, 6, 9)
                for field in ("score_novelty", "score_feasibility")
            ],
            error="",
            model="gemini-3.1-pro-preview",
        )
    )
    appendix = "\n".join(_provenance_appendix(record))

    assert appendix.count("novelty score") == 1
    assert (
        "novelty score 8 → 4, 6 → 3, and 9 → 4; feasibility score 8 → 4, 6 → 3, "
        "and 9 → 4 (answered on a 1-10 scale)." in appendix
    )
    # And it says why one field carries three readings, rather than leaving a
    # reader to decide whether the record can be pinned to an idea.
    assert (
        "The repair pass records the field it repaired and not which idea's copy "
        "of it, so the values above are in the order it met them." in appendix
    )


def test_a_title_never_opens_a_bracket_it_does_not_close():
    """Claims carry their parameters in brackets, so a nine-word cut lands inside
    one often enough to matter: "... Island Coating (5 Nm" was a real heading."""
    title = derive_idea_title(
        "A sacrificial nanoscale zinc-oxide (ZnO) island coating (5 nm, chemical "
        "vapor deposition) extends the cycle life of the cathode"
    )
    assert title == "A Sacrificial Nanoscale Zinc-oxide (ZnO) Island Coating"


def test_a_bracket_the_cut_did_not_reach_is_left_alone():
    title = derive_idea_title(
        "Applying a 10 nm Lithium Lanthanum Titanium Oxide (LLTO) coating via "
        "sol-gel deposition raises cycle count"
    )
    assert title == "Applying a 10 nm Lithium Lanthanum Titanium Oxide (LLTO)"


def test_a_method_name_cut_after_its_preposition_is_not_kept_in_part():
    """ "Applied via Atomic" named no method: the rest was atomic layer deposition.

    Nor does "Applied" alone, once the method is gone -- applied to what, by what --
    so the title stops at the last thing the cut left whole.
    """
    title = derive_idea_title(
        "A 2 nm Al2O3 surface coating applied via Atomic Layer Deposition "
        "suppresses structural fatigue in Ni-rich cathodes"
    )
    assert title == "A 2 nm Al2O3 Surface Coating"


def test_a_noun_that_merely_ends_in_ing_is_not_mistaken_for_a_gerund():
    """Dropping "Coating" here would leave the title without its subject."""
    title = derive_idea_title(
        "A conformal alumina coating raises the cycle count of Ni-rich cathodes "
        "well past the uncoated control"
    )
    assert title == "A Conformal Alumina Coating Raises the Cycle Count"


def test_every_dimension_reviewed_reaches_the_report(body: str):
    """Each specialist numbers its own set from rev_001, so an id-keyed dedupe
    read the novelty, feasibility, impact and safety reviews of an idea as
    repeats of its correctness review. A live run printed "No feasibility
    review was recorded" under every idea that had one."""
    assert "No feasibility review was recorded" not in body
    assert "No safety review was recorded" not in body
    assert "No novelty review was recorded" not in body
    assert "across five reviews" in body


def test_a_verdict_does_not_repeat_the_closing_turn_it_summarises(body: str):
    """The judge states its rationale in the last turn, so printing it again
    underneath repeated a full paragraph verbatim on every debated match."""
    verdicts = [
        line for line in body.splitlines() if line.startswith("The judge ruled this")
    ]
    assert verdicts
    assert "Rationale: Rationale" not in body
    for line in verdicts:
        assert "Rationale:**" not in line
        if "[Rematch" in line:
            assert line.index("[Rematch") < line.index("Rationale:")


def test_the_competing_readings_are_counted_rather_than_hedged_over(body: str):
    """ "It competes with at least one other reading of the same situation." hedged
    about a count the very next clause supplies, under every idea in the report."""
    facts = {
        "Mechanism and rationale": "The coating scavenges HF.",
        "Discriminating predictions": "Cycle life rises by a tenth.",
        "Alternative explanations": "The effect is thermal; and the control is "
        "confounded.",
        "Falsifier": "No rise over the matched control.",
    }
    one, two = (
        _idea_description(facts, alternatives)[2]
        for alternatives in (["The effect is thermal."], ["A.", "B."])
    )

    assert one.startswith("One competing reading of the same situation has to be")
    assert two.startswith("Two competing readings of the same situation have to be")
    assert "at least one other reading" not in body


def test_a_verdict_whose_reasoning_is_the_turn_above_it_points_at_nothing(body: str):
    """Pointing at the paragraph directly overhead was twelve copies of one sentence.
    Silence carries it, but only because the case with nothing to point at says so."""
    closing = "Rationale: the coated design isolates the mechanism."
    match = SimpleNamespace(
        outcome="win",
        confidence=0.75,
        debate_turns=[f"Turn 4: Judge: {closing}"],
        rationale="the coated design isolates the mechanism",
    )
    silent = SimpleNamespace(**{**vars(match), "rationale": ""})

    assert _verdict_line(match) == "The judge ruled this a win with confidence 0.75."
    assert "No rationale was recorded for it." in _verdict_line(silent)
    assert "Its rationale is the closing statement above" not in body
    # And the preamble says what an unadorned verdict means, once.
    assert body.count("its reasoning is that closing turn rather than missing") == 1


def test_a_verdict_matches_the_closing_turn_as_the_reader_was_shown_it(body: str):
    """The suppression above read the turn through ``readable_turn``, which is not
    the text on the page: the transcript is printed contribution by contribution and
    the closing rationale is given the sentence capital that its "Rationale:" label
    had kept off it. One letter apart, four live matches printed the paragraph twice
    -- "- **Turn 5, Closing rationale:** This idea provides a more feasible,
    mode-appropriate, and well-supported experimental approach. ..." and directly
    under it "The judge ruled this a win with confidence 0.75. Rationale: This idea
    provides a more feasible, mode-appropriate, and well-supported experimental
    approach. ..."

    And it read that turn even in the chapter that does not print it. A pair that
    meets twice is transcribed under one idea and cross-referenced under the other,
    where "The verdict below is how it went for this idea." stood above "The judge
    ruled this a loss with confidence 0.70." and no reason at all.
    """
    reason = "This idea provides a more feasible experimental approach."
    match = SimpleNamespace(
        outcome="win",
        confidence=0.75,
        # As ``_sided`` leaves it: the label is a proper noun that opened the
        # sentence, and the replacement takes no capital after a colon.
        debate_turns=[
            "Turn 5: Expert A: It is settled. "
            f"Rationale: {reason[:1].lower()}{reason[1:]}"
        ],
        rationale=reason,
    )

    assert _verdict_line(match) == "The judge ruled this a win with confidence 0.75."
    assert reason in _verdict_line(match, transcript_above=False)

    lines = body.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("The judge ruled this"):
            continue
        above = [text for text in lines[max(0, index - 3) : index] if text.strip()]
        if not above or "Closing rationale:**" not in above[-1]:
            continue
        closing = above[-1].split("Closing rationale:**", 1)[1].strip()
        assert closing not in line


def test_a_complete_set_of_responses_is_not_announced_before_it_is_listed(body: str):
    """ "Every review of this idea recorded a response." stood above a list naming each
    of those reviews, under all eight ideas. A count short of the reviews is the case
    the list cannot show, so that one survives."""
    reviews = [
        SimpleNamespace(section=section, rebuttals=["It holds."])
        for section in ("Correctness", "Novelty")
    ]
    silent = [SimpleNamespace(section="Feasibility", rebuttals=[])]

    assert _attributed_responses(reviews).startswith("The correctness review answered:")
    assert _attributed_responses([*reviews, *silent]).startswith(
        "Two reviews of this idea recorded a response."
    )
    assert "Every review of this idea recorded a response" not in body


def test_what_the_go_no_go_tests_are_for_is_said_once_above_the_ideas(body: str):
    """Fourteen words framing each of the eight lists, above the tests themselves."""
    assert "Whether to continue is decided against" not in body
    assert body.count("Its go/no-go tests:") > 1
    assert body.count("continuing or abandoning is decided against") == 1


def test_that_no_prediction_was_tested_is_stated_of_the_run_not_of_each_idea(
    body: str,
):
    """The run proposes work rather than doing any, so it is true of every prediction
    in the report and was printed under each idea whose case rested on one."""
    assert "which no result in this run has tested" not in body
    assert body.count("No prediction anywhere in this report has been tested") == 1


def test_a_title_does_not_stop_on_a_participle_that_needs_a_complement():
    """ "Interphase (CEI) Composed" stops one preposition short of saying anything."""
    title = derive_idea_title(
        "A conformal artificial cathode-electrolyte interphase layer (CEI) composed "
        "of fluorinated self-assembled monolayers blocks solvent co-intercalation"
    )
    assert title == "A Conformal Artificial Cathode-electrolyte Interphase Layer (CEI)"


def test_a_title_does_not_stop_on_an_adjective_whose_noun_was_cut():
    title = derive_idea_title(
        "Dry-coating NCM811 cathodes with a 1 wt% Al2O3 protective layer raises "
        "capacity retention"
    )
    assert title == "Dry-coating NCM811 Cathodes with a 1 wt% Al2O3"


def test_a_title_does_not_stop_on_a_verb_whose_object_was_cut():
    title = derive_idea_title(
        "A durable self-healing microcapsule-embedded polyurethane surface barrier "
        "coating extends calendar life at 45 C"
    )
    assert title == (
        "A Durable Self-healing Microcapsule-embedded Polyurethane Surface Barrier "
        "Coating"
    )


def test_a_noun_ending_in_ous_or_ive_is_not_mistaken_for_an_adjective():
    """The suffix rule only fires on a truncated title, so a short one is safe."""
    assert derive_idea_title(
        "Porous ZrO2 outperforms a dense conformal alternative"
    ) == ("Porous ZrO2 Outperforms a Dense Conformal Alternative")


def test_clustered_titles_are_the_subject_of_their_sentence_not_a_sentence():
    from coscientist.narrative import _join

    subject = _join(["A Porous ZrO2 Coating", "A LiF Nanoshell"], fallback="").rstrip(
        "."
    )
    assert not subject.endswith(".")
    assert f"{subject} converge on one mechanism".count(". converge") == 0


def test_a_two_item_series_of_clean_clauses_joins_on_a_comma():
    """ "; and" between two clauses reads as though an item had gone missing."""
    from coscientist.narrative import _join

    joined = _join(
        [
            "Sacrificial scavenging generates byproducts that trigger gas evolution "
            "(swelling).",
            "Anode poisoning by dissolved Zn2+ ions.",
        ],
        fallback="",
    )

    assert "; and" not in joined
    assert joined.endswith("(swelling), and anode poisoning by dissolved Zn2+ ions.")


def test_a_comma_that_cannot_be_read_as_the_separator_does_not_force_semicolons():
    """An aside, a thousands separator and a trailing clause are not item breaks."""
    from coscientist.narrative import _join

    aside = _join(
        [
            "Precise control of coating thickness (e.g., via ALD) to exactly 2 nm.",
            "Accurate EIS measurements to isolate charge transfer resistance.",
        ],
        fallback="",
    )
    assert "; and" not in aside
    assert "to exactly 2 nm, and accurate EIS" in aside

    # The comma is behind the conjunction by the time the reader meets it, so it
    # cannot be mistaken for the break between the two items.
    trailing = _join(
        [
            "Randomization and blinding can be incorporated into the protocol.",
            "A 50% effect size is large, meaning N=12 is adequately powered.",
        ],
        fallback="",
    )
    assert "; and" not in trailing
    assert "into the protocol, and a 50% effect size" in trailing

    counted = _join(
        ["The cycler logs 1,200 cycles.", "The teardown follows."], fallback=""
    )
    assert "; and" not in counted


def test_an_item_that_opens_a_colon_is_not_folded_into_a_series():
    """A colon reaches to the full stop, so a conjunct behind one joins the wrong list.

    "A direct measurement of the prediction that separates it from the field: coated
    cells will reach 80% retention 10% later than controls, and evidence for the
    competing reading its ranking assumes away" read as two things the measurement
    would show, rather than as the third and fourth entries in the list of evidence
    that would change the decision.
    """
    from coscientist.narrative import _join

    quoted = _join(
        [
            "A direct measurement of the prediction: coated cells reach 80% retention "
            "10% later than controls.",
            "Evidence for the competing reading, which is that the coating raises "
            "initial impedance.",
        ],
        fallback="",
    )
    assert "later than controls. Evidence for the competing reading" in quoted
    assert "and evidence for the competing reading" not in quoted

    # A comma that does sit between two clauses still gets the semicolon it needs:
    # ", and" there would read as a third thing the first clause detects.
    ambiguous = _join(
        [
            "The coating converts to ZnF2 over time, detectable by XRD.",
            "Electrolyte HF concentration falls.",
        ],
        fallback="",
    )
    assert "detectable by XRD; and electrolyte HF concentration falls" in ambiguous


def test_a_series_whose_members_carry_commas_keeps_its_semicolons():
    from coscientist.narrative import _join

    joined = _join(
        [
            "A 50% effect size is large, meaning N=12 is adequately powered.",
            "Blinding can be incorporated.",
            "The cost is justified, given the information gain.",
        ],
        fallback="",
    )

    assert joined.count("; ") == 2
    assert "; and the cost is justified" in joined


def test_an_item_that_is_itself_two_sentences_is_set_on_its_own():
    """Folded in, its second sentence leaves "; and ..." hanging off a fresh clause."""
    from coscientist.narrative import _join

    joined = _join(
        [
            "Randomization can be incorporated.",
            "Safety protocols manage the precursor risks. Thermal abuse tests are "
            "included as go/no-go criteria.",
            "These controls can be added during revision.",
        ],
        fallback="",
    )

    assert "go/no-go criteria; and these controls" not in joined
    assert joined == (
        "Randomization can be incorporated. Safety protocols manage the precursor "
        "risks. Thermal abuse tests are included as go/no-go criteria. These "
        "controls can be added during revision."
    )


def test_an_abbreviation_is_not_mistaken_for_the_end_of_a_sentence():
    """ "e.g. ZnO" is a stop followed by a capital, and it is not a sentence break."""
    from coscientist.narrative import _join

    joined = _join(
        ["A porous coating (e.g. ZnO) scavenges HF.", "Dense coatings do not."],
        fallback="",
    )

    assert (
        joined == "A porous coating (e.g. ZnO) scavenges HF, and dense coatings do not."
    )


def test_an_item_carrying_its_own_semicolon_does_not_extend_the_series():
    from coscientist.narrative import _join

    joined = _join(
        ["Coverage is partial; the anode side is untested.", "The cost is bounded."],
        fallback="",
    )

    assert "; and" not in joined


def test_a_list_of_nothing_but_stubs_falls_back_rather_than_printing_them():
    from coscientist.narrative import _join

    assert _join(["N/A", "none", "  "], fallback="None recorded.") == "None recorded."


def test_an_unstated_field_is_not_introduced_as_though_it_were_stated():
    """ "It cannot start until its inputs exist: no external dependency was recorded."

    Every field a summary subsection draws on is optional, and each one has a fallback
    saying so. Printed after a lead-in that promises the field, the fallback answers
    the sentence that introduced it and the pair says nothing.
    """
    from coscientist.narrative import _idea_facts, _summary_sections

    bare = SimpleNamespace(
        claim="A coating extends cycle life.",
        rationale="It blocks the electrolyte.",
        mechanism_model="",
        validation_protocol="",
        predictions=[],
        alternatives=[],
        falsifier="",
        dependencies=[],
        risks=[],
        go_no_go_tests=[],
    )
    sections = _summary_sections(
        _idea_facts(bare), [], rank=1, elo=1500, shortlisted=False
    )
    prose = " ".join(sections.values()).lower()
    for promise in (
        "its inputs exist:",
        "set down in advance:",
        "what would falsify it:",
        "that mechanism predicts:",
    ):
        assert promise not in prose, (
            f"an unstated field was introduced as stated: {promise}"
        )
    # The absence is stated outright rather than left as a silence.
    assert "neither a go/no-go threshold nor a falsifier" in prose
    assert "no discriminating prediction was stated for it" in prose


def test_a_novelty_score_is_placed_against_the_field_that_was_scored():
    """ "The novelty review scored this 5 of five." under eight of eight ideas.

    The figure is already in the review table a few lines below and in the average in
    section one, so printing it alone adds nothing. What is not anywhere else, and is
    what the score is for, is where it sits: five of five in a field topping out at
    three is a different statement from five of five where half the ideas scored five.
    """
    from coscientist.narrative import IdeaReview, _novelty_standing

    def review(score: int) -> IdeaReview:
        return IdeaReview(
            section="Novelty",
            lead_in="Novelty reviewer:",
            question="Is this new?",
            findings=[],
            objections=[],
            rebuttals=[],
            answer="It is new.",
            score=score,
        )

    top = _novelty_standing(review(5), [5, 5, 4, 2])
    assert "the highest any idea in this run received" in top
    assert "shared with one other idea" in top

    alone = _novelty_standing(review(2), [5, 5, 4, 2])
    assert "the lowest any idea in this run received, and no other matched it." in alone

    middle = _novelty_standing(review(4), [5, 5, 4, 2])
    assert "inside a field running from 2 to 5." in middle

    # A field with no spread separates nothing, and saying "the highest" of a score
    # every idea received reads as a distinction the run did not draw.
    flat = _novelty_standing(review(5), [5, 5, 5])
    assert "highest" not in flat
    assert "separates this one from none of them" in flat

    assert _novelty_standing(None, [5, 4]).startswith("No novelty review")


def test_a_check_list_is_introduced_by_where_its_items_came_from():
    """The standing checks were appended to every idea, three identical paragraphs
    under the objections that were the reason to read the section. They belong only
    where the reviews raised nothing, and the lead-in has to say which case it is."""
    from coscientist.narrative import (
        DEEP_VERIFICATION_LEAD_IN,
        DEEP_VERIFICATION_STANDING_LEAD_IN,
        IdeaReview,
        _deep_verification,
        _idea_facts,
    )

    facts = _idea_facts(
        SimpleNamespace(
            claim="A coating extends cycle life.",
            rationale="It blocks the electrolyte.",
            mechanism_model="",
            validation_protocol="Ten cells per arm against an uncoated control.",
            predictions=["Coated cells outlast uncoated cells by 15%."],
            alternatives=["The gain comes from the binder, not the coating."],
            falsifier="No difference at N=10 per arm.",
            dependencies=["An ALD reactor."],
            risks=["The coating cracks."],
            go_no_go_tests=["Thickness within 2 nm by TEM."],
        )
    )
    review = IdeaReview(
        section="Novelty",
        lead_in="Novelty reviewer:",
        question="Is this new?",
        findings=[],
        objections=["The mechanism is asserted rather than measured."],
        rebuttals=[],
        answer="Yes.",
        score=4,
    )
    lead_in, checks = _deep_verification([review], facts)
    assert lead_in == DEEP_VERIFICATION_LEAD_IN
    titles = [title for title, _ in checks]
    assert "Independent Confirmation of the Mechanism" not in titles
    assert len(checks) == 1

    lead_in, checks = _deep_verification([], facts)
    assert lead_in == DEEP_VERIFICATION_STANDING_LEAD_IN
    assert next(title for title, _ in checks) == (
        "Independent Confirmation of the Mechanism"
    )


def test_a_colon_inside_a_debate_turn_does_not_open_a_new_sentence():
    """The label a debater writes is a proper noun and keeps its capital after a colon;
    the phrase substituted for it is not one, and printed a capital mid-sentence."""
    from coscientist.narrative import _sided

    assert (
        _sided("The weakness is this: Hypothesis 1 concedes the point.", first=True)
        == "The weakness is this: this idea concedes the point."
    )
    assert (
        _sided("Hypothesis 2 is weaker.", first=True) == "The opposing idea is weaker."
    )


def test_a_gap_restated_by_a_second_specialist_is_printed_once():
    from coscientist.narrative import _without_restatements

    kept = _without_restatements(
        [
            "Empirical data demonstrating the electrochemical stability of "
            "self-healing polymers at voltages >4.0V.",
            "Missing empirical data on the electrochemical stability of self-healing "
            "polymers at high cathode voltages (>4.0V).",
        ]
    )
    assert kept == [
        "Empirical data demonstrating the electrochemical stability of "
        "self-healing polymers at voltages >4.0V."
    ]


def test_two_gaps_that_merely_share_vocabulary_are_both_printed():
    from coscientist.narrative import _without_restatements

    kept = _without_restatements(
        [
            "Full-text verification of discovered sources to confirm the exact "
            "parameters each one reports.",
            "Lack of verified evidence for the exact thickness threshold at which "
            "kinetic penalties begin.",
        ]
    )
    assert len(kept) == 2


def test_a_report_with_cited_findings_does_not_claim_nothing_was_consulted(body: str):
    consulted = "No external body of knowledge was consulted"
    assert consulted not in body or "## References\n\nNone" in body


def test_a_title_does_not_end_on_a_lone_modifier_left_after_its_preposition():
    """ "... Extension by Metal" was written about metal oxide coatings."""
    title = derive_idea_title(
        "The primary mechanism of cycle life extension by metal oxide coatings is "
        "chemical scavenging rather than physical isolation"
    )
    assert title == "The Primary Mechanism of Cycle Life Extension"


def test_a_prepositional_phrase_that_names_something_is_kept_whole():
    title = derive_idea_title(
        "A 10 nm double-layer coating of Al2O3 nanoparticles and a conductive "
        "polymer improves thermal stability"
    )
    assert title == "A 10 nm Double-layer Coating of Al2O3 Nanoparticles"


def test_a_source_id_quoted_by_the_manifest_is_named_not_printed(
    rich_session: Session,
):
    """The manifest quotes sources by id the way every other specialist does."""
    from coscientist.models import Artifact

    rich_session.artifacts.append(
        Artifact(
            stage="report",
            agent="dossier",
            artifact_type="specialist_output",
            content="",
            schema_name="DossierManifest",
            payload={
                "title": "Protective coatings for lithium-ion cycle life",
                "sections": [],
                "evidence_that_would_change_decision": [
                    "Full-text verification of source_1 to confirm the parameters "
                    "it reports"
                ],
            },
        )
    )
    body = compile_dossier(rich_session).split(_APPENDIX)[0]
    assert "source_1" not in body
    assert "Full-text verification of the source" in body
    _assert_no_record_ids(body)


def test_clustered_idea_titles_keep_the_case_the_report_prints_them_in():
    from coscientist.narrative import _joined_titles

    subject = _joined_titles(
        ["A Defect-free ZrO2 Coating", "An Artificial Interphase (CEI)"]
    )
    assert subject == "A Defect-free ZrO2 Coating and An Artificial Interphase (CEI)"
    assert not subject.endswith(".")


def test_a_pair_of_titles_is_joined_without_the_list_semicolon():
    """The semicolon reads as though a third item were coming."""
    from coscientist.narrative import _joined_titles

    assert _joined_titles(["One Idea", "Another Idea"]) == "One Idea and Another Idea"
    assert _joined_titles(["A", "B", "C"]) == "A; B; and C"


def test_a_long_first_name_that_holds_an_and_of_its_own_gets_its_end_marked():
    """A live description cited two papers as "the unverified claims drawn from
    Identification of the dual roles of Al2O3 coatings on NMC811-cathodes via theory
    and experiment and Tailoring Performance of the LiNi0.8Mn0.1Co0.1O2 Cathode by
    Al2O3 and MoO3 artificial cathode electrolyte interphase (CEI) layers through
    plasma-enhanced atomic layer deposition (PEALD) Coating" -- three "and"s across
    the join, and nothing saying which of them ends the first name.

    Long and broken, both: a short pair the reader can still hold whole needs no mark,
    and the semicolon on one reads as though a third were coming."""
    from coscientist.narrative import _joined_titles

    marked = _joined_titles(
        [
            "Identification of the dual roles of Al2O3 coatings on NMC811-cathodes "
            "via theory and experiment",
            "Tailoring Performance of the LiNi0.8Mn0.1Co0.1O2 Cathode by Al2O3 and "
            "MoO3 artificial CEI layers",
        ]
    )

    assert marked.startswith("Identification of the dual roles")
    assert "via theory and experiment; and Tailoring Performance" in marked
    # Short enough to be read whole, conjunction and all.
    assert _joined_titles(
        ["negative or null results", "corrections or retractions"]
    ) == ("negative or null results and corrections or retractions")
    # Long, but with nothing inside it that could be mistaken for the join.
    assert _joined_titles(
        ["A Defect-free ZrO2 Coating Applied by Atomic Layer Deposition", "Another"]
    ) == ("A Defect-free ZrO2 Coating Applied by Atomic Layer Deposition and Another")


@pytest.mark.parametrize(
    ("stated", "folded"),
    [
        ("ZnO dissolves into the electrolyte", "ZnO dissolves into the electrolyte"),
        ("LiF coating increases impedance", "LiF coating increases impedance"),
        ("EIS shows a resistance rise", "EIS shows a resistance rise"),
        ("The barrier effect dominates", "the barrier effect dominates"),
        ("Sol-gel coatings leave pits", "sol-gel coatings leave pits"),
        ("A porous shell outperforms it", "a porous shell outperforms it"),
    ],
)
def test_a_folded_sentence_keeps_the_case_a_formula_needs(stated: str, folded: str):
    from coscientist.narrative import _spliced

    assert _spliced(stated) == folded


def test_a_unit_symbol_survives_headline_casing():
    """ "Nm" is a newton-metre; the claim said nanometre."""
    assert derive_idea_title("a 5 nm LiF nanoshell raises retention") == (
        "A 5 nm LiF Nanoshell Raises Retention"
    )
    assert derive_idea_title("dry-coating at 1 wt% raises retention") == (
        "Dry-coating at 1 wt% Raises Retention"
    )


def test_a_debate_turn_is_read_from_the_side_whose_section_prints_it():
    """The same transcript appears twice; half of it used to name the wrong idea."""
    from coscientist.narrative import _sided

    turn = "Hypothesis 1 is stronger. Hypothesis 2 relies on H1's own mechanism."

    assert _sided(turn, first=True) == (
        "This idea is stronger. The opposing idea relies on this idea's own mechanism."
    )
    assert _sided(turn, first=False) == (
        "The opposing idea is stronger. This idea relies on the opposing idea's own "
        "mechanism."
    )


def test_an_idea_named_by_its_position_is_re_sided_like_one_named_by_number():
    """A judge who writes "the first proposal" names a slot, not an idea.

    "Avoids the material instability that plagues the first proposal" was reprinted
    unchanged on both ideas' pages -- once beside "this idea" and once beside "the
    opposing idea" -- and neither page says which proposal was presented first.
    """
    from coscientist.narrative import _sided

    turn = "The first proposal is weaker. The latter avoids the instability."

    assert _sided(turn, first=True) == (
        "This idea is weaker. The opposing idea avoids the instability."
    )
    assert _sided(turn, first=False) == (
        "The opposing idea is weaker. This idea avoids the instability."
    )


def test_a_strategy_enum_quoted_by_a_debater_is_written_as_words(body: str):
    for enum in ("competing_explanation", "evidence_first", "mechanism_first"):
        assert enum not in body


def test_the_deep_verification_caveat_is_stated_once_in_the_whole_report(body: str):
    """It was printed under every objection, forty-one times in one report. Moved to
    the section lead-in it was down to eight -- one per idea, of a caveat that holds
    for all of them -- and from there into the preamble above the ideas, where it is
    stated once. Each item names its own review, so the only lead-in a list still
    needs is the one for a list nobody raised anything into."""
    dives = body.count("#### Deep Verification")
    assert dives > 1
    assert body.count("which objection each one reaches is left to the reader") == 1
    assert body.count("so each is a live claim against the idea") == 1
    standing = body.count("so what follows is the standing check")
    assert standing < dives, "the fixture must give some idea a raised objection"
    # Every other list opens straight onto its first numbered item.
    opened = re.findall(r"#### Deep Verification\n\n(.+)", body)
    assert len(opened) - standing == sum(
        line.startswith("##### 1. ") for line in opened
    )


def test_an_objection_is_printed_once_per_idea_and_counted_under_its_review(
    rich_session: Session, body: str
):
    """Every objection used to be printed twice inside one idea: once as a bullet under
    the review that raised it and again under Deep Verification. Deep Verification is
    the copy that carries the raising review and whether anything answered it, so the
    review keeps only the count. Where that copy is printed is a fact about every idea
    in the report and belongs in the preamble; under each review it was forty copies of
    one signpost."""
    briefs = build_idea_briefs(load_record(rich_session))
    dives = _idea_sections(body)
    assert len(dives) == len(briefs)

    assert "Objections raised:" not in body
    assert "printed under Deep Verification below" not in body
    assert "each numbered and attributed to the review that raised it" in body
    for brief, dive in zip(briefs, dives, strict=True):
        raised = {
            objection for review in brief.reviews for objection in review.objections
        }
        assert raised, "the fixture must give every idea something to object to"
        for objection in raised:
            assert dive.count(objection) == 1, f"printed twice in one idea: {objection}"
        # The count still stands where the reader meets the review that raised it.
        assert re.search(r"This review raised \w+ objections?\.", dive)
        assert "#### Deep Verification" in dive


def test_the_count_under_a_review_is_hoisted_where_no_review_differs_on_it(body: str):
    """The count distinguishes a review that answered from one that stood on its
    objection, and on a live run of eight ideas none of them differed: "This review
    raised one objection and recorded one response." stood between the findings and
    the score of all five reviews of all eight, forty times over."""
    reviews = [
        SimpleNamespace(
            objections=["It is thin."], rebuttals=["It is not."], stood_in=False
        )
        for _ in range(40)
    ]
    briefs = [SimpleNamespace(reviews=reviews[index::8]) for index in range(8)]

    said, hoisted = shared_review_tally(briefs)
    assert hoisted
    assert said[0].startswith(
        "Each of the forty reviews below raised one objection and recorded one "
        "response, and that is the same under every idea"
    )

    # One review that answered nothing is the case the sentence exists to mark, so
    # the count goes back under each review rather than being said over all of them.
    quiet = [*reviews[:-1], SimpleNamespace(**{**vars(reviews[-1]), "rebuttals": []})]
    assert shared_review_tally([SimpleNamespace(reviews=quiet)]) == ([], False)
    # A placeholder raised nothing at all, and is not a reviewer to speak for.
    stood_in = [
        *reviews[:-1],
        SimpleNamespace(**{**vars(reviews[-1]), "stood_in": True}),
    ]
    assert shared_review_tally([SimpleNamespace(reviews=stood_in)]) == ([], False)
    # The fixture's reviews do differ, so its report still carries the count in place.
    assert "This review raised" in body


def test_a_response_is_printed_once_per_idea_and_counted_under_its_review(
    rich_session: Session, body: str
):
    """The responses were the objections' defect one section earlier.

    "Rebuttals offered:" under each review printed the same sentences the same idea's
    Addressed Objections had already printed, attributed. The attributed copy survives:
    a response written with its subject elided -- "could be useful for verifying the
    exact thickness dependence, but fundamentally lacks novelty" -- is only readable
    where the review that wrote it is named. The review keeps the count, which is the
    one thing the summary cannot show: that it answered rather than stood pat.
    """
    briefs = build_idea_briefs(load_record(rich_session))
    dives = _idea_sections(body)

    assert "Rebuttals offered:" not in body
    counted = 0
    for brief, dive in zip(briefs, dives, strict=True):
        answered = {
            rebuttal for review in brief.reviews for rebuttal in review.rebuttals
        }
        assert answered, "the fixture must give every idea a recorded response"
        for rebuttal in answered:
            assert dive.count(rebuttal) == 1, f"printed twice in one idea: {rebuttal}"
        counted += len(
            re.findall(
                r"This review (?:raised \w+ objections? and )?recorded \w+ ", dive
            )
        )
    assert counted, "the count has to stand where the reader meets the review"


def test_a_deep_verification_item_states_the_objection_and_stops(body: str):
    """The pairing caveat belongs in the lead-in; per item it was pure boilerplate."""
    for boilerplate in (
        "That review did record responses",
        "That review recorded no response of any kind",
    ):
        assert boilerplate not in body


def test_a_cluster_mechanism_is_stated_once_and_not_in_three_sections():
    """Main Research Directions named every cluster with its mechanism; the research
    direction bullets reprinted the same pairs a few hundred words later; and every
    converging pair under Unexpected Connections printed one of them a third time."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord

    mechanism = "A coating suppresses the interfacial reaction."
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_a": "A Thin Alumina Coating",
        "cand_b": "A Thin Zirconia Coating",
    }
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism=mechanism,
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )

    overview = synthesize_overview(record)
    printed = "\n".join(
        [
            *(
                paragraph
                for section in overview.sections
                for paragraph in section.paragraphs
            ),
            *overview.research_directions,
            *overview.unexpected_connections,
        ]
    )

    assert printed.lower().count(mechanism.rstrip(".").lower()) == 1
    assert "Physical Barrier Coatings" in overview.unexpected_connections[0]


def test_an_idea_placed_under_two_mechanisms_is_said_to_be_counted_twice():
    """A live report opened "three distinct clusters", gave their sizes as two, two
    and one over four ideas, and printed the same hypothesis under two converging
    pairs further down -- each pair described as two ideas resting on one shared
    mechanism. Nothing said the sizes double-count it."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _section_three

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_a": "A Thin Alumina Coating",
        "cand_b": "A Thin Zirconia Coating",
        "cand_c": "A Doped Cathode Surface",
    }
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism="A coating suppresses the interfacial reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            ),
            ResearchCluster(
                name="Surface Chemistry",
                candidate_ids=["cand_b", "cand_c"],
                shared_mechanism="Surface chemistry sets the reaction rate.",
                shared_outcome="Retention holds past five hundred cycles.",
            ),
        ]
    )

    printed = "\n".join(_section_three(record).core)

    assert "One idea appears in more than one cluster: A Thin Zirconia Coating." in (
        printed
    )
    assert "count it once per cluster" in printed
    assert "total more than the number of ideas mapped" in printed

    record.landscape.clusters[1].candidate_ids = ["cand_a", "cand_b"]
    both = "\n".join(_section_three(record).core)
    assert "Two ideas appear in more than one cluster:" in both
    assert "count each of them once per cluster" in both

    record.landscape.clusters[1].candidate_ids = ["cand_c"]
    assert "more than one cluster" not in "\n".join(_section_three(record).core)


def test_an_idea_under_no_cluster_at_all_is_named_rather_than_left_out():
    """The paragraph opens "Mapping the generated ideas back onto the problem" and a
    live run gave its cluster sizes as two, one and one over a field of eight --
    clustering had run on the shortlist. A reader totalling the clusters counts half
    the ideas, and the closing sentence about what a shared mechanism costs a
    portfolio silently does not reach the other half."""
    from coscientist.models import (
        Candidate,
        CandidatePopulation,
        ResearchCluster,
        ResearchLandscape,
    )
    from coscientist.narrative import ResearchRecord, _section_three

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_a": "A Thin Alumina Coating",
        "cand_b": "A Thin Zirconia Coating",
        "cand_c": "A Doped Cathode Surface",
    }
    record.population = CandidatePopulation(
        candidates=[
            Candidate(
                id=name,
                title=record.titles[name],
                claim=f"{name} raises retention.",
                rationale="Because the coating blocks the reaction.",
                mechanism_model="The coating blocks the reaction that drives fade.",
                validation_protocol="Coin cells against an uncoated control.",
                falsifier="Retention does not improve.",
            )
            for name in ("cand_a", "cand_b", "cand_c")
        ]
    )
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism="A coating suppresses the interfacial reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )

    printed = "\n".join(_section_three(record).core)

    assert (
        "One idea was not placed under any of this cluster: A Doped Cathode Surface."
        in printed
    )
    assert "is not said of it" in printed

    # And a run where the clustering reached every idea says nothing at all.
    record.landscape.clusters[0].candidate_ids = ["cand_a", "cand_b", "cand_c"]
    assert "not placed under any of" not in "\n".join(_section_three(record).core)


def test_an_overlapping_idea_is_named_by_the_title_the_report_gave_it():
    """Clustering runs after evolution and may name a revision. A live report said
    "one idea appears in more than one cluster: A 2.5 nm ALD-deposited LiNbO3 Coating
    Improves Cycle Life" -- a title derived from the rewritten claim and printed
    nowhere else in the document."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _section_three

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_a": "A Thin Alumina Coating",
        "cand_b": "A Thin Zirconia Coating",
        "rev_b": "A Thin Zirconia Coating Improves Cycle Life",
    }
    record.lineage = {"rev_b": "cand_b"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism="A coating suppresses the interfacial reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            ),
            ResearchCluster(
                name="Surface Chemistry",
                candidate_ids=["rev_b"],
                shared_mechanism="Surface chemistry sets the reaction rate.",
                shared_outcome="Retention holds past five hundred cycles.",
            ),
        ]
    )

    printed = "\n".join(_section_three(record).core)

    assert "One idea appears in more than one cluster: A Thin Zirconia Coating." in (
        printed
    )
    assert "Improves Cycle Life" not in printed


def test_a_cluster_holding_an_idea_and_its_own_revision_counts_one_idea():
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _section_three

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {"cand_a": "A Thin Alumina Coating", "rev_a": "A Thicker Coating"}
    record.lineage = {"rev_a": "cand_a"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "rev_a"],
                shared_mechanism="A coating suppresses the interfacial reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )

    printed = "\n".join(_section_three(record).core)

    assert "Physical Barrier Coatings holds one idea" in printed
    assert "more than one cluster" not in printed


def test_a_shared_mechanism_written_as_a_clause_is_introduced_by_a_colon():
    """ "holds one idea around conformal LiAlF4 coating provides a physical barrier
    against HF attack" -- the clustering stage writes the mechanism as a whole clause
    as often as it writes a noun phrase, and "around" only fits the second."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _section_three

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {"cand_a": "A Thin Alumina Coating"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a"],
                shared_mechanism="A coating blocks the electrolyte from the surface.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )

    printed = "\n".join(_section_three(record).core)

    assert (
        "holds one idea, grouped on this mechanism: a coating blocks the electrolyte "
        "from the surface." in printed
    )
    assert " around a coating blocks" not in printed


def test_open_questions_does_not_reprint_the_recommendation_it_follows():
    """Section 9 lists the evidence that would change the recommendation, and Open
    Questions took the same list from the same artifact and printed it again as
    bullets a page later -- on a live run, three of its four bullets were section
    nine's sentences word for word and the fourth was a rewording of one of them."""
    from coscientist.models import DossierManifest, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _open_questions

    decisive = "Full-text verification of the coating thickness reported in source 3."
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.manifest = DossierManifest(
        title="Ranked ideas",
        sections=[],
        evidence_that_would_change_decision=[decisive],
    )
    record.landscape = ResearchLandscape(
        coverage_gaps=[
            # The landscape's own rewording of the same gap, which is how it got
            # back into the list once the verbatim copy was taken out.
            "Lack of verified full-text evidence confirming the coating thickness "
            "reported in source 3.",
            "No cell-level cost estimate was retrieved for any coating route.",
        ]
    )

    questions, lead_in = _open_questions(record)

    assert questions == [
        "No cell-level cost estimate was retrieved for any coating route."
    ]
    assert "Recommendations and Next Steps above" in lead_in
    assert "One further item was recorded here and is not listed" in lead_in


def test_a_finding_already_printed_is_not_reprinted_as_something_unknown(
    rich_session: Session,
):
    """A discovery pass writes its uncertainties off the findings it reports, so two
    of a live report's seven open questions were its own Main Research Directions
    reworded -- a finding restated under a heading saying the run does not know it."""
    from coscientist.models import Artifact, DiscoveryManifest, DiscoveryNarrative
    from coscientist.narrative import ResearchRecord, _open_questions

    finding = (
        "Independent replication has produced contradictory evidence regarding "
        "complete coverage of ALD, with clear signals of exposed surface Lithium "
        "even after 10 full ALD deposition cycles."
    )
    reworded = (
        "While ALD is praised for its self-limiting conformity, independent "
        "replication has produced contradictory evidence regarding complete "
        "coverage, discovering clear signals of exposed surface Lithium even after "
        "10 full ALD deposition cycles."
    )
    session = Session(question="Does a coating help?")
    session.artifacts = [
        Artifact(
            stage="evidence",
            agent="discovery_agent",
            artifact_type="specialist_output",
            content="",
            schema_name="DiscoveryManifest",
            payload=DiscoveryManifest(
                question="Does a coating help?",
                narratives=[
                    DiscoveryNarrative(
                        question="Does a coating help?",
                        statements=[
                            DiscoveryStatement(
                                text=reworded,
                                facet="contradictory",
                                originating_pass=1,
                            )
                        ],
                        uncertainties=[
                            finding,
                            "No cell-level cost estimate was retrieved.",
                        ],
                    )
                ],
            ).model_dump(),
        )
    ]
    record = ResearchRecord(session=session)
    record.discovery = DiscoveryManifest.model_validate(session.artifacts[0].payload)

    questions, lead_in = _open_questions(record)

    assert questions == ["No cell-level cost estimate was retrieved."]
    assert "restates a finding already printed under Main Research Directions" in (
        lead_in
    )


def test_open_questions_says_so_rather_than_inventing_one_when_nothing_is_left():
    from coscientist.models import DossierManifest
    from coscientist.narrative import ResearchRecord, _open_questions

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.manifest = DossierManifest(
        title="Ranked ideas",
        sections=[],
        evidence_that_would_change_decision=["A direct measurement of the thickness."],
    )

    questions, lead_in = _open_questions(record)

    assert lead_in == ""
    assert questions == [
        "Nothing was left open beyond the evidence named under Recommendations and "
        "Next Steps above."
    ]


def test_the_facets_the_pass_scored_zero_on_are_one_bullet_and_not_seven():
    """Seven bullets differing by one noun each, two of them ungrammatical.

    A gap against a facet the discovery pass scores is one of a fixed set of seven, so
    a run that scored zero on all of them printed the set as seven open questions --
    including "No adequate negative null evidence was discovered.", the enum name with
    its underscores swapped for spaces. The facets are named from FACET_PHRASES here
    rather than from the recorded description, because a session written before that
    mapping existed carries the enum spelling in the description.
    """
    from coscientist.models import (
        EVIDENCE_FACETS,
        DiscoveryCoverage,
        DiscoveryManifest,
        ResearchGap,
    )
    from coscientist.narrative import ResearchRecord, _open_questions

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.discovery = DiscoveryManifest(
        question="Can a coating help?",
        coverage_history=[
            DiscoveryCoverage(
                gaps=[
                    ResearchGap(
                        direction="Evidence landscape",
                        facet=facet,
                        description=(
                            f"No adequate {facet.replace('_', ' ')} evidence was "
                            "discovered."
                        ),
                    )
                    for facet in EVIDENCE_FACETS
                ]
            )
        ],
    )

    questions, _ = _open_questions(record)
    assert questions == [
        "The discovery pass found no adequate evidence under any facet it scores: "
        "supporting evidence, evidence contradicting the leading direction, negative "
        "or null results, independent replication, methodological detail, safety or "
        "governance evidence, and corrections or retractions affecting the sources "
        "used."
    ]

    record.discovery.coverage_history[0].gaps = [
        gap
        for gap in record.discovery.coverage_history[0].gaps
        if gap.facet in {"negative_null", "methods"}
    ]
    partial, _ = _open_questions(record)
    assert partial == [
        "The discovery pass found no adequate evidence under two facets it scores: "
        "negative or null results and methodological detail."
    ]


def test_a_normalizer_complaint_is_not_printed_as_a_scientific_open_question():
    """ "No citation-linked statements could be normalized." opened a live report's
    Open Questions, above seven questions about the field -- a bullet about the
    report's own machinery, and the one entry in the section no experiment could
    close. The producer stopped writing it; a session recorded before that still
    holds it, and this is where those are read."""
    from coscientist.models import DiscoveryManifest, DiscoveryNarrative
    from coscientist.narrative import ResearchRecord, _open_questions

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.discovery = DiscoveryManifest(
        question="Can a coating help?",
        narratives=[
            DiscoveryNarrative(
                question="Can a coating help?",
                summary="A report.",
                uncertainties=[
                    "No citation-linked statements could be normalized.",
                    "Cycle life past one thousand cycles is unmeasured.",
                ],
                disagreements=[
                    "The extractor returned no JSON matching the schema.",
                    "Two groups report opposite thickness optima.",
                ],
            )
        ],
    )

    questions, _ = _open_questions(record)

    assert questions == [
        "Cycle life past one thousand cycles is unmeasured.",
        "Two groups report opposite thickness optima.",
    ]


def test_a_template_landscapes_coverage_gaps_are_not_this_runs_open_questions():
    """Three noun-phrase labels the clustering fallback carries, read as findings.

    "Negative and null-result evidence." and "External-validity boundary conditions."
    are the deterministic fallback's own defaults; set among questions written about
    the goal they read as gaps this run found, and they are bare labels where every
    other bullet in the section is a stated sentence.
    """
    from coscientist.models import ResearchLandscape
    from coscientist.narrative import ProvenanceNote, ResearchRecord, _open_questions

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.landscape = ResearchLandscape(
        coverage_gaps=[
            "Independent replication evidence",
            "External-validity boundary conditions",
        ]
    )

    stated, _ = _open_questions(record)
    assert stated == [
        "Independent replication evidence.",
        "External-validity boundary conditions.",
    ]

    record.provenance = [
        ProvenanceNote(
            stage="proximity",
            agent="proximity_agent",
            schema_name="ResearchLandscape",
            source="deterministic_fallback",
            repairs=[],
            error="",
        )
    ]
    templated, _ = _open_questions(record)
    assert templated == [
        "The clustering stage fell back to a template, so what it recorded as "
        "coverage gaps is that template's default list rather than anything this run "
        "found missing, and it is not stated as an open question here."
    ]


def test_a_coverage_gap_is_written_in_words_rather_than_in_enum_tokens():
    """The facet is an enum, and the gap description printed it with its underscores
    swapped for spaces -- "No adequate corrections retractions evidence"."""
    from coscientist.evidence import audit_coverage
    from coscientist.models import DiscoveryNarrative

    coverage = audit_coverage(
        DiscoveryNarrative(question="Can a coating help?", summary="", statements=[]),
        leads=[],
        previous=None,
    )

    descriptions = " ".join(gap.description for gap in coverage.gaps)
    assert "corrections retractions" not in descriptions
    assert "negative null" not in descriptions
    assert "safety governance" not in descriptions
    assert "corrections or retractions" in descriptions


def test_cross_links_are_listed_in_the_order_the_ideas_were_ranked_in():
    """Every other list of ideas is in tournament order and this one was in whatever
    order the clustering pass emitted, so the ideas a reader met as first and third
    came back as third and first and read as a different pair."""
    from coscientist.models import ResearchCluster, ResearchLandscape, TournamentState
    from coscientist.narrative import ResearchRecord, _unexpected_connections

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_1": "First",
        "cand_2": "Second",
        "cand_3": "Third",
        "cand_4": "Fourth",
    }
    record.tournament = TournamentState(
        ratings={"cand_1": 1600.0, "cand_2": 1550.0, "cand_3": 1500.0, "cand_4": 1450.0}
    )
    record.landscape = ResearchLandscape(
        clusters=[
            # Emitted worst-first, and each with its members the wrong way round.
            ResearchCluster(
                name="Low Cluster",
                candidate_ids=["cand_4", "cand_3"],
                shared_mechanism="A slow mechanism.",
                shared_outcome="A slow outcome.",
            ),
            ResearchCluster(
                name="High Cluster",
                candidate_ids=["cand_2", "cand_1"],
                shared_mechanism="A fast mechanism.",
                shared_outcome="A fast outcome.",
            ),
        ]
    )

    connections, counts = _unexpected_connections(record)

    assert counts.converging == 2
    assert connections[0].startswith("First and Second")
    assert connections[1].startswith("Third and Fourth")


def test_the_lead_in_does_not_call_a_near_duplicate_pair_something_other_than_a_pair():
    """It promised pairs, counted the clusters, and called everything else "not pairs"
    -- of which a flagged near-duplicate is the clearest pair in the section."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import (
        ResearchRecord,
        _unexpected_connections,
        connections_lead_in,
    )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {"cand_a": "Alumina", "cand_b": "Zirconia", "cand_c": "Zirconia II"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism="A coating suppresses the reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ],
        duplicates=[["cand_b", "cand_c"]],
    )

    connections, counts = _unexpected_connections(record)
    lead_in = connections_lead_in(counts)

    assert (counts.converging, counts.duplicates, counts.minority) == (1, 1, 0)
    assert len(connections) == 2
    assert "not pairs" not in lead_in
    assert "near-duplicate" in lead_in
    # The protected-minority clause is the one that genuinely is not about a pair,
    # and this run has no protected minority to hang it on.
    assert "protected minority" not in lead_in


def test_the_lead_in_tells_a_sole_occupant_apart_from_a_thinly_shared_region():
    """It described every protected entry as a region held open by one idea, six
    lines above an entry naming the other idea in that same region."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import (
        ResearchRecord,
        _unexpected_connections,
        connections_lead_in,
    )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {"cand_a": "Alumina", "cand_b": "Zirconia", "cand_c": "Titania"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_b", "cand_c"],
                shared_mechanism="A coating suppresses the reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ],
        # One sole occupant and one sharing a two-idea region.
        protected_minority_ids=["cand_a", "cand_b"],
    )

    _connections, counts = _unexpected_connections(record)
    lead_in = connections_lead_in(counts)

    assert (counts.sole_minority, counts.shared_minority) == (1, 1)
    assert "rests on a single idea" in lead_in
    assert "more than one occupant" in lead_in

    record.landscape.protected_minority_ids = ["cand_a"]
    _, sole_only = _unexpected_connections(record)
    assert (sole_only.sole_minority, sole_only.shared_minority) == (1, 0)
    sole_lead_in = connections_lead_in(sole_only)
    assert "more than one occupant" not in sole_lead_in
    # One entry, so the clause that introduces it is singular. "The remaining entries
    # ... They are about how thinly a region is covered" stood over a single bullet.
    assert "The remaining entry is" in sole_lead_in
    assert "The remaining entries" not in sole_lead_in


def test_the_connections_lead_in_does_not_promise_a_mechanism_the_run_never_recorded():
    """It sent the reader to Main Research Directions for "the mechanism its cluster
    is named for", which on a fallback clustering is one filler sentence copied into
    every cluster -- and which section three, correctly, declines to print."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import (
        ResearchRecord,
        _unexpected_connections,
        connections_lead_in,
    )

    filler = "Candidates share a generation lens but retain distinct predictions."
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_a": "Alumina",
        "cand_b": "Zirconia",
        "cand_c": "Titania",
        "cand_d": "Silica",
    }
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Evidence First",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism=filler,
                shared_outcome="Retention holds past five hundred cycles.",
            ),
            ResearchCluster(
                name="Mechanism First",
                candidate_ids=["cand_c", "cand_d"],
                shared_mechanism=filler,
                shared_outcome="Retention holds past five hundred cycles.",
            ),
        ]
    )

    _connections, counts = _unexpected_connections(record)
    lead_in = connections_lead_in(counts)

    assert not counts.named_mechanisms
    assert "the mechanism its cluster is named for" not in lead_in
    assert "mechanism that tells its clusters apart" in lead_in
    assert "stand or fall on the same claim" not in lead_in
    # The consequence is about a cluster, whatever size the cluster came out.
    assert "a pair below shares a label" not in lead_in
    assert "cluster below groups its ideas under a label" in lead_in


def test_a_lone_cluster_of_three_is_not_introduced_as_two_ideas():
    """ "What this section reports is where two ideas rest on a single mechanism" is
    the size a cluster happens to be most often, not the size this run found. Over a
    lone cluster of three it undercounts the exposure the sentence exists to name."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import (
        ResearchRecord,
        _unexpected_connections,
        connections_lead_in,
    )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {"cand_a": "Alumina", "cand_b": "Zirconia", "cand_c": "Titania"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b", "cand_c"],
                shared_mechanism="A coating suppresses the reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )

    _connections, counts = _unexpected_connections(record)
    lead_in = connections_lead_in(counts)

    assert (counts.converging, counts.converging_members) == (1, 3)
    assert "where three ideas rest on a single mechanism" in lead_in
    assert "two ideas rest" not in lead_in


def test_the_cost_of_sharing_a_cluster_is_not_stated_over_a_cluster_of_one():
    """A live run printed "two ideas in the same cluster fail for the same reason, so
    funding both buys less information than the pair of scores would suggest" over four
    clusters of which two held a single idea each. A cluster of one has no such pair in
    it, and the lead-in above promised "the mechanism its members share" for it too."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _section_three

    def cluster(name: str, *ids: str) -> ResearchCluster:
        return ResearchCluster(
            name=name,
            candidate_ids=list(ids),
            shared_mechanism=f"The {name.lower()} route sets the rate.",
            shared_outcome="Retention holds past five hundred cycles.",
        )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {f"cand_{item}": item.upper() for item in "abcd"}
    record.landscape = ResearchLandscape(
        clusters=[
            cluster("Barrier", "cand_a", "cand_b"),
            cluster("Doping", "cand_c"),
            cluster("Electrolyte", "cand_d"),
        ]
    )

    printed = "\n".join(_section_three(record).core)

    assert "the mechanism its members share" not in printed
    assert "the mechanism it was grouped on" in printed
    assert "two ideas in the same cluster fail for the same reason" in printed
    assert "That bears on one cluster above; the other two clusters hold one idea" in (
        printed
    )

    record.landscape.clusters[0].candidate_ids = ["cand_a"]
    only_singletons = "\n".join(_section_three(record).core)
    assert "No cluster here holds more than one idea" in only_singletons
    assert "fail for the same reason" not in only_singletons
    assert "each idea stands or falls on its own" in only_singletons


def test_a_retrieved_finding_is_printed_once_and_pointed_at_afterwards(
    rich_session: Session, body: str
):
    """Comparison with Existing Solutions opened by reprinting the first three
    findings verbatim, under a report that had already printed all of them."""
    from coscientist.narrative import _evidence_statements

    statements = _evidence_statements(load_record(rich_session))
    assert statements, "the fixture retrieved no finding to print"
    # Section 7 only. Elsewhere a finding is quoted to a purpose -- section 4 names
    # the one claim an idea cites that was recorded as contradicting it -- and the
    # duplication was between the two sections that printed the whole set as a set.
    comparison = body[body.index("\n#### 7. ") : body.index("\n#### 8. ")]
    for statement in statements:
        text = " ".join(statement.text.rstrip(".").split())
        assert text not in comparison, f"a finding is reprinted: {text[:70]!r}"
    assert "Main Research Directions above" in comparison, (
        "the comparison drops the findings without saying where they are"
    )


def test_the_evidence_integrity_list_runs_in_the_order_the_ideas_are_ranked_in(
    rich_session: Session,
):
    """It ran in candidate-id order, which is an order the reader is never shown.

    The two cases that carry no per-idea detail are now one line each naming the ideas
    they cover, so rank order is what orders the names inside a line.
    """
    from coscientist.narrative import evidence_integrity_lines

    record = load_record(rich_session)
    lines = evidence_integrity_lines(record)
    assert len(lines) > 1, "the fixture flags fewer than two groundings"
    ranked = [brief.title for brief in build_idea_briefs(record)]
    for line in lines:
        listed = [title for title in ranked if title in line]
        assert listed, f"a line names no idea: {line}"
        assert listed == sorted(listed, key=line.index)
    flagged = [title for title in ranked for line in lines if title in line]
    assert len(flagged) == len(set(flagged)), "an idea is flagged under two cases"


def test_the_integrity_list_names_the_withdrawn_paper_rather_than_its_id(
    rich_session: Session,
):
    """A live appendix read "cites evidence that was retracted or could not be
    retrieved: claim_6_1, source_6_2" -- the one section of the report about evidence
    that cannot be trusted, and the one place in it that never said what the evidence
    was. The record is held, so which paper was withdrawn is knowable here."""
    from coscientist.narrative import evidence_integrity_lines

    (line,) = [
        item
        for item in evidence_integrity_lines(load_record(rich_session))
        if "discredited" in item
    ]

    assert "claim_3" not in line
    assert "the retracted claim drawn from Cycle-life benchmarking" in line


def test_an_id_the_session_cannot_place_is_set_as_the_identifier_it_is(
    rich_session: Session,
):
    """Nothing names it, so it is printed -- in code font, so a reader can see that
    what they are being shown is a literal identifier and not a mangled word."""
    from coscientist.narrative import evidence_integrity_lines

    (line,) = [
        item
        for item in evidence_integrity_lines(load_record(rich_session))
        if "does not exist in this session" in item
    ]

    assert "`claim_missing`" in line


def test_a_reference_the_report_never_cites_is_not_listed(rich_session: Session):
    """A number is claimed before the paragraph holding it can still be cut."""
    report = compile_dossier(rich_session)
    entries = _reference_entries(report.split(_APPENDIX)[0])
    cited = {
        int(number)
        for group in re.findall(r"\[(\d+(?:, \d+)*)\]", report)
        for number in group.split(", ")
    }
    assert cited == set(range(1, len(entries) + 1))


def test_the_scoring_legend_does_not_read_one_verdict_off_an_ambiguous_number(
    rich_session: Session,
):
    """The legend read "a printed four is therefore a confidently held revise and a
    printed two a confidently held rejection" and then, in the same sentence, that
    neither can be read off the number alone. A four is equally an advance its own
    reviewer was diffident about, so the example refuted the rule it illustrated."""
    report = compile_dossier(rich_session)

    assert "confidently held revise and a" not in report
    assert (
        "a four is a confidently held revise or an advance its reviewer was "
        "diffident about" in report
    )
    assert "Which of them a given score is has to be read off the review itself" in (
        report
    )


def test_recommending_two_ideas_from_one_cluster_says_what_that_costs():
    """The report said clustered ideas fail together and then recommended two out of
    one cluster, leaving four recommendations reading as four independent bets."""
    from coscientist.models import DossierManifest, ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _recommended_overlaps

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {
        "cand_a": "A Thin Alumina Coating",
        "cand_b": "A Thin Zirconia Coating",
        "cand_c": "A Solvent-free Dry Coating",
    }
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism="A coating suppresses the interfacial reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )
    record.manifest = DossierManifest(
        title="Ranked Research Ideas",
        sections=[],
        recommendation_candidate_ids=["cand_a", "cand_c", "cand_b"],
    )

    assert _recommended_overlaps(record, ["cand_a", "cand_c", "cand_b"]) == [
        (
            "Physical Barrier Coatings",
            ["A Thin Alumina Coating", "A Thin Zirconia Coating"],
        )
    ]

    prose = " ".join(
        paragraph
        for section in synthesize_overview(record).sections
        if section.number == 9
        for paragraph in section.paragraphs
    )
    assert "Part of what is recommended is one bet rather than several" in prose
    assert "A Thin Alumina Coating and A Thin Zirconia Coating sit in the" in prose
    assert "A Solvent-free Dry Coating sit in the" not in prose
    # "So they stand or fall together" was printed over a pair whose two hypotheses
    # were that ultrathin coatings do not improve cycle life and that a 2.5 nm
    # coating does. They cannot stand together and they cannot fall together: the
    # shared mechanism is the thing they disagree about.
    assert "stand or fall together" not in prose
    assert "Sharing a mechanism is not agreeing about it" in prose
    assert "one result on it decides more than one idea on this list at once" in prose


def test_ideas_that_finished_level_are_not_printed_as_an_ordering():
    """Three ideas on 1184 were ranked four, five and six by the sort's tie-break."""
    from coscientist.narrative import IdeaBrief

    def brief(rank: int, tied_with: int) -> IdeaBrief:
        return IdeaBrief(
            title="",
            candidate_id="",
            rank=rank,
            elo=1184,
            category="",
            proposal="",
            description=[],
            facts={},
            summary={},
            table_rows=[],
            reviews=[],
            coherence=[],
            deep_verification=[],
            matches=[],
            wins=0,
            losses=0,
            ties=0,
            shortlisted=False,
            tied_with=tied_with,
        )

    assert brief(1, 0).rank_line == "Rank: 1"
    assert brief(4, 1).rank_line == (
        "Rank: 4, shared on Elo with another idea and listed arbitrarily among them"
    )
    assert brief(4, 2).rank_line == (
        # Spelled, as every other small count in prose is.
        "Rank: 4, shared on Elo with two other ideas and listed arbitrarily among them"
    )


def test_the_standings_are_a_grid_and_level_ideas_share_a_position(
    rich_session: Session, body: str
):
    """ "The full standings run A at 1290 (6-0); B at 1234 (4-2); ..."

    Eight ideas with a rating and a record apiece is a table, and as a sentence it was
    ninety words a reader had to hold in mind to compare any two of them. Numbering the
    rows 1..8 would also report an order the tournament did not produce, so ideas that
    finished level share a position.
    """
    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    five = next(
        section
        for section in synthesize_overview(record).sections
        if section.number == 5
    )
    assert "The full standings run" not in " ".join(five.paragraphs)
    grid = next(iter(five.grids))
    assert grid.columns[:3] == ["Position", "Idea", "Elo"]
    assert [row[1] for row in grid.rows] == [brief.title for brief in briefs]
    # The paragraphs after it read off the standings, so it prints where it is
    # introduced rather than at the end of the section.
    assert five.paragraphs[grid.after].startswith("The full standings follow")
    assert "| Position | Idea | Elo |" in body

    by_elo: dict[str, set[str]] = {}
    for position, _title, elo, *_ in grid.rows:
        by_elo.setdefault(elo, set()).add(position)
    for elo, positions in by_elo.items():
        assert len(positions) == 1, f"ideas level on {elo} were printed as an ordering"


def test_a_level_idea_is_given_the_same_position_wherever_the_report_states_it(
    rich_session: Session,
):
    """Table 9 folded three ideas on 1184 onto position 4 and section four called the
    same three ideas rank 4, rank 5 and rank 6 "by sort order" -- two numberings of one
    tournament, four pages apart, with nothing saying which the run produced."""
    from coscientist.narrative import _section_four, _shared_positions

    # A block of ties takes the position of its first member and the block after it
    # resumes at its own place in the listing, which is what the tournament separated.
    assert _shared_positions(
        [("cand_a", 1290), ("cand_b", 1184), ("cand_c", 1184), ("cand_d", 1100)]
    ) == {"cand_a": 1, "cand_b": 2, "cand_c": 2, "cand_d": 4}

    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    grid = next(
        iter(
            next(
                section
                for section in synthesize_overview(record).sections
                if section.number == 5
            ).grids
        )
    )
    stated = {row[1]: row[0] for row in grid.rows}
    assert stated, "the standings table printed no rows to agree with"
    written = {
        section.title: " ".join(section.paragraphs)
        for section in _section_four(record, briefs).subsections
    }
    for brief in briefs:
        prose = written[brief.title]
        position = stated[brief.title]
        # Every idea states its standing one way or the other; which one depends on
        # whether it finished level with anything.
        assert (
            f"shares position {position} with" in prose
            or f"finished rank {position} on an Elo" in prose
        ), f"{brief.title} is position {position} in the standings but not in prose"


def test_the_evidence_subsection_names_the_findings_the_idea_cites(
    rich_session: Session, body: str
):
    """A heading promising evidence, under which no finding was ever named.

    The subsection held two sentences about what a discriminating prediction is
    worth, printed under all eight ideas with the idea's own cited findings nowhere
    in the report except the evidence appendix.
    """
    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    candidates = {candidate.id: candidate for candidate in record.candidates}
    quoted = 0
    for brief in briefs:
        slot = brief.summary["Supporting Arguments & Evidence (Motivation)"]
        assert slot in body
        cited = set(candidates[brief.candidate_id].evidence_ids)
        supporting = [
            claim
            for claim in record.evidence.claims
            if claim.id in cited and claim.relation == "supports"
        ]
        for claim in supporting:
            assert claim.claim.rstrip(".") in slot, (
                f"{brief.title} cites {claim.id} and the subsection headed with the "
                "word evidence does not say so"
            )
        quoted += bool(supporting)
        if not supporting:
            assert "The findings this idea cites" not in slot
        # The evidence stage classifies a finding against the research question, so
        # the report cannot say the idea's own citations argue for the idea.
        assert "in support of itself" not in slot
    assert 0 < quoted < len(briefs), (
        "the fixture no longer covers both an idea that cites evidence and one that "
        "does not"
    )


def test_the_conclusion_names_the_review_that_scored_the_idea_lowest(
    rich_session: Session, body: str
):
    """The order of work is the same sentence for every idea that recorded both a
    threshold and a falsifier, and it was the whole of the section: eight conclusions
    in two variants, none of them about the idea above them. Where the idea is weakest
    is on the record and is not stated anywhere else -- the reviews are printed in
    section order and the lead-in to them gives the span without the dimension."""
    briefs = build_idea_briefs(load_record(rich_session))
    for brief in briefs:
        conclusion = brief.summary["Conclusion"]
        assert conclusion in body
        scores = [review.score for review in brief.reviews]
        assert min(scores) < max(scores), "the fixture no longer varies the scores"
        floor = min(scores)
        # Read only the clause about the floor: "Feasibility Assessment" is named by
        # the order-of-work sentence above it, so a search of the whole conclusion
        # finds the word feasibility whatever the feasibility review scored.
        assert "lowest score" in conclusion
        clause = conclusion[conclusion.index("lowest score") :]
        assert f"{floor} of five" in clause
        for review in brief.reviews:
            assert (review.section.lower() in clause) == (review.score == floor), (
                f"the {review.section} review scored {review.score} against a floor "
                f"of {floor} and the conclusion says otherwise: {clause}"
            )


def test_the_conclusion_does_not_send_a_reader_to_a_review_nobody_wrote(
    rich_session: Session,
):
    """The worst place the substitution can surface, and where it did: the rank-1
    idea's lowest score was the placeholder's, so the sentence naming what to read
    before commissioning the work named a review no reviewer had written."""
    from dataclasses import replace

    from coscientist.narrative import _conclusion, _idea_facts

    record = load_record(rich_session)
    facts = _idea_facts(record.candidates[0])
    reviews = [
        replace(review, score=4) for review in build_idea_briefs(record)[0].reviews
    ]
    reviews[0] = replace(reviews[0], score=2, stood_in=True)

    said = _conclusion(facts, reviews, shortlisted=True, accepted_flaw=None)

    assert f"from the {reviews[0].section.lower()} review" in said
    assert "That score is a placeholder's rather than a reviewer's" in said
    assert "unreviewed rather than weak" in said


def test_a_lowest_score_a_reviewer_did_set_down_is_not_called_a_placeholders(
    rich_session: Session,
):
    from dataclasses import replace

    from coscientist.narrative import _conclusion, _idea_facts

    record = load_record(rich_session)
    facts = _idea_facts(record.candidates[0])
    reviews = [
        replace(review, score=4) for review in build_idea_briefs(record)[0].reviews
    ]
    reviews[0] = replace(reviews[0], score=2)

    said = _conclusion(facts, reviews, shortlisted=True, accepted_flaw=None)

    assert "placeholder" not in said


def test_the_conclusion_says_what_the_next_move_cannot_settle(rich_session: Session):
    """The three cases the live sessions do not reach, and the punctuation of a pair."""
    from dataclasses import replace

    from coscientist.narrative import AdjudicationNote, _conclusion, _idea_facts

    record = load_record(rich_session)
    facts = _idea_facts(record.candidates[0])
    reviews = build_idea_briefs(record)[0].reviews
    assert len(reviews) > 2

    level = [replace(review, score=4) for review in reviews]
    assert "no one dimension of it is weaker" in _conclusion(
        facts, level, shortlisted=True, accepted_flaw=None
    )

    pair = [replace(review, score=4) for review in reviews]
    pair[0] = replace(pair[0], score=2)
    pair[1] = replace(pair[1], score=2)
    shared = _conclusion(facts, pair, shortlisted=True, accepted_flaw=None)
    names = f"{pair[0].section.lower()} and {pair[1].section.lower()}"
    # "the correctness, and feasibility reviews" punctuates two items as though a
    # third had been dropped between them.
    assert f"shared by the {names} reviews" in shared

    accepted = _conclusion(
        facts,
        reviews,
        shortlisted=True,
        accepted_flaw=AdjudicationNote(
            candidate_id=record.candidates[0].id,
            title="",
            resolution="override",
            adjudicator="the run owner",
            justification="",
            fatal_flaws=["The protocol cycles cells outside their rated window."],
        ),
    )
    # A test returns a verdict on the idea. It cannot return one on a flaw a person
    # allowed to stand, so a conclusion that stops at the bench work reads as a plan
    # that disposes of the objection above it.
    assert "None of that work reaches the fatal flaw accepted above" in accepted

    # An idea the shortlist dropped has no next move. Asserting one and withdrawing it
    # two clauses later -- "The next move is the go/no-go work ... It is not on the
    # shortlist, so nothing is scheduled against it" -- was the conclusion of four of
    # eight ideas in a live report.
    dropped = _conclusion(facts, reviews, shortlisted=False, accepted_flaw=None)
    assert dropped.startswith("It is not on the shortlist, so nothing is scheduled")
    assert "If it is revived, the first move is" in dropped
    assert "The next move is" not in dropped
    # Why the go/no-go precedes the falsifier is true of every idea that records both
    # and is stated once in the preamble above the ideas, not eight times here.
    assert "cheaper of the two" not in dropped


def test_the_fatal_review_band_names_the_exceptions_rather_than_relisting(
    rich_session: Session,
):
    """Six of seven titles listed, then listed again, is the same wall of names twice.

    The sentence after the affected-ideas list says which of them drew more than one
    such review. Where that is nearly all of them, naming the ones it is not says the
    same thing in a line; where it is all of them, no list is needed at all.
    """
    from dataclasses import replace

    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    assert len(briefs) >= 3

    def _band(counts: list[int]) -> str:
        """The band paragraph, with each of the first ideas holding ``counts`` fatals.

        Only that paragraph: the leading idea is named again where the section says
        which proposal won, and that mention is not the repetition under test.
        """
        adjusted = []
        for index, brief in enumerate(briefs):
            fatal = counts[index] if index < len(counts) else 0
            reviews = [
                replace(review, score=2 if position < fatal else 4)
                for position, review in enumerate(brief.reviews)
            ]
            adjusted.append(replace(brief, reviews=reviews))
        paragraph = next(
            block
            for block in _section_eight(record, adjusted).core
            if "closed at two or below" in block
        )
        return " ".join(paragraph.split())

    lone = briefs[2].title
    mixed = _band([2, 2, 1])
    assert f"All of them except {lone.rstrip('.')} drew more than one" in mixed
    # The exception is the only title the second sentence repeats; the other two are
    # named once, in the list of affected ideas above it.
    assert mixed.count(briefs[0].title) == 1
    assert mixed.count(briefs[1].title) == 1

    every = _band([2, 2, 2])
    assert "Every one of them drew more than one" in every
    assert every.count(briefs[2].title) == 1

    # One idea over the band still has to be named: there is no set to except it from.
    single = _band([2])
    assert f"{briefs[0].title.rstrip('.')} drew more than one" in single

    # Where the band takes the whole field, the list excludes nothing and is seven long
    # titles to say what one clause says -- before the next sentence names most of them
    # again as the ones that drew more than one.
    whole = _band([2] * len(briefs))
    assert "No idea in the run escaped the band." in whole
    assert "The affected ideas are" not in whole
    assert whole.count(briefs[1].title) == 0


def test_an_exclusion_that_takes_the_whole_field_is_not_spelled_out_idea_by_idea(
    rich_session: Session,
):
    """Seven long titles to say what one clause says, and none of them excluded."""
    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    record.manifest = DossierManifest(
        title="Dossier",
        sections=[],
        unresolved_fatal_flaw_candidate_ids=[brief.candidate_id for brief in briefs],
    )
    paragraph = next(
        block
        for block in _section_eight(record, briefs).core
        if "from any recommendation" in block
    )
    assert paragraph.startswith(
        "The meta-review excluded every ranked idea from any recommendation, stating "
        "that an unresolved fatal flaw stands against each."
    )
    for brief in briefs:
        assert brief.title not in paragraph


def test_a_check_that_found_nothing_reports_that_rather_than_going_silent(
    rich_session: Session,
):
    """ "Whether the reviews carry the flaw it names is checked below." then nothing.

    The two verification paragraphs are emitted only on a mismatch, so on a run where
    the reviews bear the meta-review out the promise was followed by the withdrawal
    paragraph and the reader was left to decide whether the check had been skipped or
    had passed.
    """
    record = load_record(rich_session)
    briefs = build_idea_briefs(record)
    # The fixture records no fatal flaw at all, and an exclusion carried by nothing is
    # the mismatch case rather than the agreement one under test.
    faulted_id = briefs[0].candidate_id
    review = next(
        item
        for review_set in record.reviews
        for item in review_set.reviews
        if record.ranked_id(item.candidate_id) == faulted_id
    )
    review.fatal_flaws = ["The coating dissolves in the electrolyte."]
    faulted = sorted(record.recorded_fatal_flaw_ids)
    assert faulted == [faulted_id]
    assert len(faulted) < len(briefs), "and must leave an idea standing"

    def _exclusion(excluded: list[str]) -> str:
        record.manifest = DossierManifest(
            title="Dossier",
            sections=[],
            unresolved_fatal_flaw_candidate_ids=excluded,
        )
        return next(
            block
            for block in _section_eight(record, briefs).core
            if "from any recommendation" in block
        )

    agreed = _exclusion(faulted)
    assert agreed.endswith(
        "The reviews carry the flaw in every case: one was recorded against each idea "
        "named here, and against no idea the meta-review left standing."
    )
    assert "is checked below" not in agreed

    unfaulted = next(
        brief.candidate_id for brief in briefs if brief.candidate_id not in faulted
    )
    disagreed = _exclusion([*faulted, unfaulted])
    assert disagreed.endswith(
        "Whether the reviews carry the flaw it names is checked below."
    )


def test_no_paragraph_is_printed_three_times_inside_one_idea(body: str):
    """Description, the summary and Deep Verification each reprinted the mechanism."""
    import difflib

    dives = _idea_sections(body)
    assert dives
    for dive in dives:
        # A per-match notice is repeated once per match by design, so the check runs
        # over the narrative half, where every paragraph is written about one idea.
        paragraphs = [
            block.strip()
            for block in dive.split("\n### Tournament\n")[0].split("\n\n")
            if len(block.strip()) > 150 and not block.startswith(("|", "-"))
        ]
        for index, later in enumerate(paragraphs):
            for earlier in paragraphs[:index]:
                overlap = difflib.SequenceMatcher(None, later, earlier).ratio()
                assert overlap < 0.75, (
                    f"a paragraph is restated at {overlap:.0%} inside one idea: "
                    f"{later[:90]!r}"
                )


def test_a_lead_with_no_title_is_named_by_the_publisher_its_link_points_at():
    """A live list printed "Untitled source lead." beside a locator on acs.org.

    An entry that names neither the document nor where to look for it is a
    reference to nothing. Discovery records a title for most leads and a bare
    link for the rest, and the link says which publisher holds the paper, so
    only a lead carrying neither now falls through to the anonymous wording.
    """
    from coscientist.narrative import _reference_title

    titled = SourceLead(
        canonical_url="https://pubs.acs.org/doi/10.1021/acsaem.3c00001",
        title="",
        year=2024,
    )
    assert _reference_title(titled) == "Untitled source on pubs.acs.org (2024)"

    redirect = SourceLead(
        canonical_url=(
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ"
        ),
        title="",
    )
    assert _reference_title(redirect) == "Untitled source lead"
    assert _reference_title(SourceLead(canonical_url="", title="")) == (
        "Untitled source lead"
    )


def test_a_judges_own_markup_does_not_reach_the_bullet_that_quotes_it():
    """A live dossier carried "**Conclusion:** Hypothesis 2 ..." inside a bullet whose
    own label is bold, so the stray asterisks closed the report's emphasis rather than
    the judge's and the rest of the line ran on in bold."""
    block = _match_block(
        _undebated_match(1, "**Conclusion:** Hypothesis 1 isolates the **mechanism**."),
        _undebated_match(2, "Turn 4: Rationale: the control is confounded."),
    )

    assert "- **Round 1 against Opponent 1 (win):** Hypothesis 1 isolates the " in block
    assert "**Conclusion" not in block
    assert "Rationale:" not in block
    assert "Turn 4" not in block


def test_a_rematch_note_is_not_printed_inside_the_reason_it_precedes():
    block = _match_block(
        _undebated_match(
            1,
            "[Rematch: this pair also met in Swiss round 1.] "
            "Hypothesis 1 isolates the mechanism.",
        ),
        _undebated_match(2, "The control is confounded."),
    )

    assert "[Rematch" not in block
    assert "Hypothesis 1 isolates the mechanism." in block


def test_a_summary_cell_is_cut_where_a_clause_ends_not_inside_one():
    """Every Falsifier Summary cell of a live summary table ended mid-condition --
    "or if the capacity retention after…" -- which states half a test and leaves the
    reader nothing to hold the idea to."""
    cell = _cell(
        "If DEMS analysis shows no significant reduction in O2 evolution during the "
        "first ten cycles, or if the capacity retention after five hundred cycles "
        "falls below eighty per cent, the oxygen-buffer mechanism is falsified."
    )

    assert cell.endswith("during the first ten cycles…")
    assert "or if" not in cell


def test_a_cell_with_no_clause_boundary_still_falls_back_to_a_word():
    cell = _cell("Retention " + "held " * 40)

    assert cell.endswith("held…")
    assert len(cell) <= 140


def test_notation_a_specialist_wrote_in_tex_is_set_as_the_text_it_stands_for():
    """Nothing downstream of the compiler renders TeX. A live report carried
    "aluminum oxide ($Al_2O_3$)" in the body and, across a page of the Knowledge
    Base, "a precisely controlled $\\mathbf{1\\text{--}5 \\text{ nm}}$ cathode
    coating" -- in the Markdown, the PDF and the DOCX alike."""
    from coscientist.dossier import _without_math_markup

    assert _without_math_markup("oxide ($Al_2O_3$) on") == "oxide (Al2O3) on"
    assert _without_math_markup(r"of $n \ge 5$ cells") == "of n ≥ 5 cells"
    assert (
        _without_math_markup(r"a $\mathbf{1\text{--}5 \text{ nm}}$ coating")
        == "a 1\u20135 nm coating"
    )
    assert (
        _without_math_markup(r"($\mathbf{1.15 \times 10^{-8} \text{ S cm}^{-1}}$)")
        == "(1.15 \u00d7 10\u207b\u2078 S cm\u207b\u00b9)"
    )
    # A caret a reader can parse beats a word lifted letter by letter.
    assert _without_math_markup("$x^{max}$ holds") == "x^max holds"


def test_two_sums_of_money_on_one_line_are_not_read_as_a_formula():
    """The provenance appendix prints what a discovery run cost."""
    from coscientist.dossier import _without_math_markup

    line = "Deep Research ran seven passes, at an estimated cost of $21.00, and"
    assert _without_math_markup(line) == line
    assert _without_math_markup("between $3.00 and $21.00 a pass") == (
        "between $3.00 and $21.00 a pass"
    )


def test_the_integrity_lead_in_agrees_with_the_number_of_ideas_it_covers(
    rich_session: Session,
):
    """Two of the four cases name every idea they cover on one line.

    So the line count says nothing about the idea count, and a run whose one
    qualified idea produced one line was given "The grounding of the following ideas
    carries a qualification" over it.
    """
    from coscientist.dossier import _provenance_appendix

    record = load_record(rich_session)
    qualified = [key for key, item in record.evidence_support.items() if item.qualified]
    assert len(qualified) > 1, "the fixture must qualify more than one idea"

    grounded = next(
        key for key, item in record.evidence_support.items() if not item.qualified
    )
    record.evidence_support = {
        key: record.evidence_support[key] for key in (qualified[0], grounded)
    }
    one = "\n".join(_provenance_appendix(record))

    assert "The following idea carries a qualification on its grounding" in one
    assert "following ideas" not in one
    assert "names the idea it applies to" in one

    # And the same count decides the all-of-them wording.
    record.evidence_support = {qualified[0]: record.evidence_support[qualified[0]]}
    every = "\n".join(_provenance_appendix(record))

    assert "The one idea in this run carries a qualification on its grounding" in every


def test_two_integrity_cases_are_separated_from_the_or_inside_one_of_them(
    rich_session: Session,
):
    """One case is itself an alternation, so joining two with a bare "or" gave the
    reader three branches and nothing marking which two are the pair: "its evidence
    was retracted or could not be retrieved or its evidence was never checked against
    its source" stood over the live report's Evidence integrity list.
    """
    from coscientist.citations import Citation
    from coscientist.dossier import _provenance_appendix
    from coscientist.models import EvidenceClaim
    from coscientist.narrative import evidence_integrity_cases

    record = load_record(rich_session)
    # Exactly the two the live report printed: the alternating case, and one more.
    record.evidence_support = {
        key: item
        for key, item in record.evidence_support.items()
        if item.support in ("discredited", "unverified")
    }
    # The alternation is written only where the run recorded both verdicts, and the
    # fixture's one discredited citation was retracted. Adding the other verdict is
    # what puts the "or" inside the case that this test is about joining.
    discredited = next(
        item
        for item in record.evidence_support.values()
        if item.support == "discredited"
    )
    discredited.citations.append(
        Citation(
            reference="claim_gone",
            claim=EvidenceClaim(
                id="claim_gone",
                claim="A coating held retention past five hundred cycles.",
                verification_status="inaccessible",
            ),
        )
    )
    assert evidence_integrity_cases(record) == [
        "its evidence was retracted or could not be retrieved",
        "its evidence was never checked against its source",
    ], "the fixture no longer produces the two cases this defect needs"

    appendix = "\n".join(_provenance_appendix(record))

    assert (
        "its evidence was retracted or could not be retrieved, or its evidence was "
        "never checked against its source." in appendix
    )
    assert "retrieved or its evidence" not in appendix


def test_the_integrity_lead_in_offers_only_the_verdicts_the_run_recorded(
    rich_session: Session,
):
    """ "Its evidence was retracted or could not be retrieved" stood over four lines
    that each named "the unretrieved claim drawn from" a source. Nothing in that run
    was retracted, and the one sentence a reader takes the case from still offered
    retraction as a live possibility -- which is different work to repair."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import evidence_integrity_cases

    record = load_record(rich_session)
    record.evidence_support = {
        key: item
        for key, item in record.evidence_support.items()
        if item.support == "discredited"
    }
    for citations in record.evidence_support.values():
        for citation in citations.citations:
            if citation.discredited:
                # Both, because a claim cannot stand better than its document: set
                # on the claim alone, a claim drawn from a retracted paper is still
                # a retraction.
                for cited in (citation.claim, citation.source):
                    if cited is not None:
                        cited.verification_status = "inaccessible"

    assert evidence_integrity_cases(record) == ["its evidence could not be retrieved"]

    appendix = "\n".join(_provenance_appendix(record))

    assert "its evidence could not be retrieved." in appendix
    assert "retracted" not in appendix


def test_the_appendix_records_which_pass_found_which_source():
    """The Knowledge Summary tells the reader that "which sources a pass found is
    recorded per pass in the discovery appendix", and the appendix gave one total for
    the whole search and no way to reach a pass from it. Every lead in the manifest
    carries the passes that returned it, so the breakdown was always on the record."""
    from coscientist.dossier import _sources_per_pass
    from coscientist.models import DiscoveryManifest
    from coscientist.narrative import ResearchRecord

    leads = [
        SourceLead(canonical_url="https://x/1", title="One", originating_passes=[1]),
        SourceLead(canonical_url="https://x/2", title="Two", originating_passes=[1, 2]),
        SourceLead(canonical_url="https://x/3", title="Three", originating_passes=[2]),
        SourceLead(canonical_url="https://x/4", title="Four"),
    ]
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.discovery = DiscoveryManifest(
        question="Can a coating help?", source_leads=leads
    )
    record.citations = CitationRegistry(leads)
    record.citations.number("https://x/2")

    said = "\n".join(_sources_per_pass(record))

    assert "- Pass 1 returned two source leads, cited in this report as [1]." in said
    assert "- Pass 2 returned two source leads, cited in this report as [1]." in said
    # No silent caps: the rows overlap and one lead is in none of them.
    assert "One lead came back from more than one pass and is counted under each" in (
        said
    )
    assert "One lead records no pass and is in no row at all." in said
    # The appendix reports the numbering rather than adding to it: asking for a
    # number would put every lead it mentions into the reference list.
    assert record.citations.numbered("https://x/1") is None
    assert len(record.citations) == 1


def test_a_run_of_one_pass_gets_no_per_pass_breakdown_of_its_own_sources():
    """One row restating the total above it is a section that says nothing."""
    from coscientist.dossier import _sources_per_pass
    from coscientist.models import DiscoveryManifest
    from coscientist.narrative import ResearchRecord

    leads = [
        SourceLead(canonical_url="https://x/1", title="One", originating_passes=[1])
    ]
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.discovery = DiscoveryManifest(
        question="Can a coating help?", source_leads=leads
    )
    record.citations = CitationRegistry(leads)

    assert not _sources_per_pass(record)


def test_an_integrity_line_does_not_close_an_ellipsis_with_a_full_stop(
    rich_session: Session,
):
    """ "...conventional carbonate …." reached a live report, in the one section it
    carries about evidence that cannot be trusted."""
    from coscientist.narrative import _stopped

    assert _stopped("conventional carbonate …") == "conventional carbonate …"
    assert _stopped("a coating helps") == "a coating helps."
    assert _stopped("a coating helps.") == "a coating helps."
    assert _stopped("does it? ") == "does it?"


def test_a_cross_link_the_report_cannot_name_is_counted_rather_than_printed():
    """The proximity stage flagged two near-duplicate pairs by ids nothing else in the
    run carries, and the section set both bullets as "The following were flagged as
    near-duplicates and should be merged before either is funded: Unnamed Research Idea
    and Unnamed Research Idea" -- a recommendation to merge two ideas it will not
    name."""
    from coscientist.models import ResearchCluster, ResearchLandscape
    from coscientist.narrative import ResearchRecord, _unexpected_connections

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.titles = {"cand_a": "Alumina", "cand_b": "Zirconia"}
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Physical Barrier Coatings",
                candidate_ids=["cand_a", "cand_b"],
                shared_mechanism="A coating suppresses the reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ],
        duplicates=[["prox_7", "prox_9"], ["prox_11", "prox_12"]],
    )

    connections, counts = _unexpected_connections(record)

    assert "Unnamed Research Idea" not in " ".join(connections)
    # Not printed, but not dropped either: the run recorded two findings and the
    # reader is told that is what became of them.
    assert counts.duplicates == 0
    assert (
        "Two further entries were recorded here against ideas this report cannot "
        "name" in connections[-1]
    )


def _rewrite_addressing_handles():
    """A rewrite whose recorded critiques are the ids evolution was thinking in."""
    from coscientist.models import Candidate, EvolutionCycle, EvolutionRecord
    from coscientist.narrative import ResearchRecord

    ranked = Candidate(
        id="cand_1",
        title="An Alumina Coating",
        claim="A coating improves cycle life.",
        rationale="It blocks the electrolyte.",
        mechanism_model="Surface passivation.",
        validation_protocol="Cycle five coated cells against five bare ones.",
        falsifier="Retention does not improve.",
    )
    rewritten = ranked.model_copy(
        update={"id": "cand_1_v2", "claim": "A 2 nm coating improves cycle life."}
    )
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.lineage = {"cand_1_v2": "cand_1"}
    record.evolution = EvolutionCycle(
        records=[
            EvolutionRecord(
                parent_ids=["cand_1"],
                candidate=rewritten,
                changes=["Specified the thickness."],
                critiques_addressed=["rev_4", "rev_cand_mechanism_2"],
                new_prediction="Retention holds past five hundred cycles.",
            )
        ]
    )
    return record, ranked


def test_a_rewrite_that_names_only_a_handle_says_it_addressed_the_reviews():
    """Evolution answered with the ids it was thinking in, and the report printed
    "evolution rewrote the idea in round one to address `rev_4`, and
    `rev_cand_mechanism_2`" -- two references that appear nowhere else in the
    document."""
    from coscientist.narrative import _revised_form

    record, ranked = _rewrite_addressing_handles()

    prose, _changed, _unchanged = _revised_form(record, ranked)

    assert "rev_4" not in prose and "rev_cand_mechanism_2" not in prose
    assert "to address the reviews" in prose


def test_naming_the_ids_does_not_hide_a_handle_from_the_test_for_one():
    """The two passes ran in the order that defeats the one above. The id-naming pass
    sets a handle it cannot place in backticks, and a backticked handle is not a bare
    one -- so the handles came back, in the branch that asserts they are what the
    rewrite was written for: "this is what it was written against: `rev_4`, and
    `rev_cand_mechanism_2`"."""
    from coscientist.narrative import _name_ids_in_prose, _revised_form

    record, ranked = _rewrite_addressing_handles()
    _name_ids_in_prose(record)

    prose, _changed, _unchanged = _revised_form(record, ranked)

    assert "rev_4" not in prose and "rev_cand_mechanism_2" not in prose
    assert "to address the reviews" in prose


def test_a_governance_finding_names_the_evidence_it_cites():
    """The naming pass walks the contracts, and the governance notes are dataclasses
    holding a copy of the same reviewer text taken off the session. One finding was
    therefore printed two ways in one report -- named where the review section quoted
    it, "relies entirely on claim_2_1" where the governance section did."""
    from coscientist.models import EvidenceClaim, EvidencePacket, SourceRecord
    from coscientist.narrative import (
        AdjudicationNote,
        BlockerNote,
        ResearchRecord,
        _name_ids_in_prose,
    )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[
            SourceRecord(
                id="source_2_2",
                url="https://example.org/alumina",
                title="Ultrathin Al2O3 Coatings",
                verification_status="verified",
            )
        ],
        claims=[
            EvidenceClaim(
                id="claim_2_1",
                claim="A 2 nm layer holds capacity past five hundred cycles.",
                source_id="source_2_2",
                verification_status="verified",
            )
        ],
    )
    record.adjudications = [
        AdjudicationNote(
            candidate_id="cand_1",
            title="An Alumina Coating",
            resolution="override",
            adjudicator="A. Reviewer",
            justification="Accepted because claim_2_1 is enough for a pilot.",
            fatal_flaws=["The idea relies entirely on claim_2_1."],
            claim="A coating improves cycle life, per source_2_2.",
        )
    ]
    record.open_governance_blocks = [
        BlockerNote(
            candidate_id="cand_2",
            title="A Titania Coating",
            fatal_flaws=["Nothing but source_2_2 speaks to the inhalation risk."],
        )
    ]

    _name_ids_in_prose(record)

    printed = " ".join(
        [
            *record.adjudications[0].fatal_flaws,
            record.adjudications[0].claim,
            *record.open_governance_blocks[0].fatal_flaws,
        ]
    )
    assert "claim_2_1" not in printed and "source_2_2" not in printed
    assert printed.count("Ultrathin Al2O3 Coatings") == 3
    # The adjudicator's own words are quoted as theirs, and the report does not edit
    # what a human wrote -- not even to make an id in it readable.
    assert (
        record.adjudications[0].justification
        == "Accepted because claim_2_1 is enough for a pilot."
    )


def test_two_records_cited_side_by_side_are_given_their_standing_once():
    """A live parenthesis read "(the unverified claim drawn from Identification of the
    dual roles of Al2O3 coatings on NMC811-cathodes via theory and experiment, the
    unverified claim drawn from Tailoring Performance of the LiNi0.8Mn0.1Co0.1O2
    Cathode ...)". The name of a claim runs to eight words before it reaches the
    title, and a specialist cites two of them in one bracket often enough that the
    reader is told twice over what standing to give what follows."""
    from coscientist.models import (
        Candidate,
        CandidatePopulation,
        EvidenceClaim,
        EvidencePacket,
        SourceRecord,
    )
    from coscientist.narrative import ResearchRecord, _name_ids_in_prose

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[
            SourceRecord(
                id=f"source_{index}",
                url=f"https://example.org/{index}",
                title=title,
                verification_status="discovered_unverified",
            )
            for index, title in enumerate(("Dual Roles of Al2O3", "Tailoring NMC811"))
        ],
        claims=[
            EvidenceClaim(
                id=f"claim_{index}",
                claim=f"Finding {index}.",
                source_id=f"source_{index}",
                verification_status="discovered_unverified",
            )
            for index in range(2)
        ],
    )
    record.population = CandidatePopulation(
        candidates=[
            Candidate(
                id="cand_1",
                title="An Alumina Coating",
                claim="A coating extends cycle life.",
                rationale="ALD scavenges HF (claim_0, claim_1).",
                mechanism_model="The barrier holds, per source_0 and source_1.",
                validation_protocol="Coin cells against an uncoated control.",
                falsifier="No difference at ten cells per arm.",
            )
        ]
    )

    _name_ids_in_prose(record)

    candidate = record.population.candidates[0]
    assert candidate.rationale == (
        "ALD scavenges HF (the unverified claims drawn from Dual Roles of Al2O3 and "
        "Tailoring NMC811)."
    )
    assert candidate.mechanism_model == (
        "The barrier holds, per the unverified sources Dual Roles of Al2O3 and "
        "Tailoring NMC811."
    )


def test_two_claims_of_one_paper_are_not_printed_as_two_papers():
    """Two claim ids drawn from one source carry one name, and a live sentence backed
    an idea with "(the unverified claim drawn from Identification of the dual roles of
    Al2O3 coatings on NMC811-cathodes via theory and experiment, the unverified claim
    drawn from Identification of the dual roles of Al2O3 coatings on NMC811-cathodes
    via theory and experiment, the claim drawn from Unexpected high power performance
    ...)" -- eighteen words twice over, reading as two papers where the run holds one,
    in a clause whose whole point was how much evidence stands behind the idea."""
    from coscientist.models import (
        Candidate,
        CandidatePopulation,
        EvidenceClaim,
        EvidencePacket,
        SourceRecord,
    )
    from coscientist.narrative import ResearchRecord, _name_ids_in_prose

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[
            SourceRecord(
                id="source_0",
                url="https://example.org/0",
                title="Dual Roles of Al2O3",
                verification_status="discovered_unverified",
            ),
            SourceRecord(
                id="source_1",
                url="https://example.org/1",
                title="Unexpected High Power Performance",
                verification_status="verified",
            ),
        ],
        claims=[
            EvidenceClaim(
                id="claim_0",
                claim="It scavenges HF.",
                source_id="source_0",
                verification_status="discovered_unverified",
            ),
            EvidenceClaim(
                id="claim_1",
                claim="It stabilises the surface.",
                source_id="source_0",
                verification_status="discovered_unverified",
            ),
            EvidenceClaim(
                id="claim_2",
                claim="It holds at rate.",
                source_id="source_1",
                verification_status="verified",
            ),
        ],
    )
    record.population = CandidatePopulation(
        candidates=[
            Candidate(
                id="cand_1",
                title="An Alumina Coating",
                claim="A coating extends cycle life.",
                rationale="It rests on the same base (claim_0, claim_1, claim_2).",
                # One name after the duplicate goes, which is the singular it always
                # was: a plural over one title reads as a list of one.
                mechanism_model="The barrier holds, per claim_0 and claim_1.",
                validation_protocol="Coin cells against an uncoated control.",
                falsifier="No difference at ten cells per arm.",
            )
        ]
    )

    _name_ids_in_prose(record)

    candidate = record.population.candidates[0]
    assert candidate.rationale == (
        "It rests on the same base (the unverified claim drawn from Dual Roles of "
        "Al2O3 and the claim drawn from Unexpected High Power Performance)."
    )
    assert candidate.mechanism_model == (
        "The barrier holds, per the unverified claim drawn from Dual Roles of Al2O3."
    )


def test_a_sentence_opening_on_a_bare_evidence_noun_is_not_read_as_an_instruction():
    """ "Evidence source_2 and source_5 demonstrate that ..." is an ellipsis while the
    ids are ids and an imperative once they are named. A live motivation ran "Evidence
    the unverified sources Improvement of the Cycling Performance and Thermal Stability
    of Lithium-Ion Cells by Double-Layer Coating of Cathode Materials with Al2O3
    Nanoparticles and Conductive Polymer and Conductive Polymer Frameworks in Silicon
    Anodes for Advanced Lithium-Ion Batteries demonstrate that double-layer Al2O3 and
    conductive polymer coatings significantly improve cycling and thermal stability."
    -- the reader is told to evidence two documents, and the verb that says otherwise
    is forty words away."""
    from coscientist.models import (
        Candidate,
        CandidatePopulation,
        EvidencePacket,
        SourceRecord,
    )
    from coscientist.narrative import ResearchRecord, _name_ids_in_prose

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[
            SourceRecord(
                id=f"source_{index}",
                url=f"https://example.org/{index}",
                title=title,
                verification_status="discovered_unverified",
            )
            for index, title in enumerate(
                ("Double-Layer Coating", "Polymer Frameworks")
            )
        ],
    )
    record.population = CandidatePopulation(
        candidates=[
            Candidate(
                id="cand_1",
                title="A Nacre-Mimetic Shield",
                claim="A double layer extends cycle life.",
                rationale="Brick and mortar dissipates stress. "
                "Evidence source_0 and source_1 demonstrate that it holds.",
                mechanism_model="Evidence source_0 demonstrates the barrier holds.",
                validation_protocol="Coin cells against an uncoated control.",
                falsifier="No difference at ten cells per arm.",
            )
        ]
    )

    _name_ids_in_prose(record)

    candidate = record.population.candidates[0]
    assert candidate.rationale == (
        "Brick and mortar dissipates stress. The unverified sources Double-Layer "
        "Coating and Polymer Frameworks demonstrate that it holds."
    )
    # And at the head of the field, where there is no full stop to read behind.
    assert candidate.mechanism_model == (
        "The unverified source Double-Layer Coating demonstrates the barrier holds."
    )


def test_naming_an_id_does_not_take_the_quotes_off_the_words_around_it():
    """The pass that names ids swept every apostrophe-and-space out of any field that
    held one, to clear the quotes it had orphaned. A live mechanism read "This
    inorganic 'brick layer is overcoated with a 2-3 nm conformal conductive polymer
    'mortar layer" -- both closing quotes eaten, in a sentence whose only id was
    somewhere else entirely."""
    from coscientist.models import Candidate, CandidatePopulation
    from coscientist.narrative import ResearchRecord, _name_ids_in_prose

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[
            Candidate(
                id="cand_nacre",
                title="A Nacre-Mimetic Shield",
                claim="A double layer holds the particles together.",
                rationale="The idea rests on 'claim_2_1' and nothing else.",
                mechanism_model=(
                    "This inorganic 'brick' layer is overcoated with a conductive "
                    "polymer 'mortar' layer, as 'cand_nacre' sets out."
                ),
                validation_protocol="Coin cells against an uncoated control.",
                falsifier="The layer cracks anyway.",
            )
        ]
    )

    _name_ids_in_prose(record)

    candidate = record.population.candidates[0]
    assert "'brick' layer" in candidate.mechanism_model
    assert "'mortar' layer" in candidate.mechanism_model
    # The quotes that were around an id come off with it: what stands there now is a
    # name this report gave the record, not anything the writer quoted.
    assert "'" not in candidate.mechanism_model.split("layer, as ")[1]
    assert (
        "'claim_2_1'" not in candidate.rationale and "'the" not in candidate.rationale
    )


def test_a_debate_turn_names_the_review_it_answers_and_not_the_id_it_is_filed_under():
    """A live transcript read "However, the `methods_statistics` review rightly points
    out a methodological flaw", and nothing else in that report calls the pass
    `methods_statistics` -- it is headed "Feasibility" and named "the methods and
    statistics review" wherever a person wrote the sentence. Panelists reach for the
    criterion ids the same way, and for the reviewer as a person, so all of those are
    covered here in the one turn."""
    from coscientist.models import PairwiseComparison, TournamentState
    from coscientist.narrative import ResearchRecord, _name_ids_in_prose

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.tournament = TournamentState(
        comparisons=[
            PairwiseComparison(
                round_number=1,
                candidate_a_id="cand_1",
                candidate_b_id="cand_2",
                presented_first_id="cand_1",
                winner_id="cand_1",
                rationale="The methods_feasibility review is answered by the protocol.",
                judge="llm_debate",
                debate_turns=[
                    "However, the `methods_statistics` review rightly points out a "
                    "methodological flaw, and the ethics_safety_governance agent and "
                    "the reflection reviewer both stand behind it.",
                    "The `safety_governance` finding is the only one left open. "
                    "Reflection also noted the risk of oxidation at high voltages. "
                    "The specular reflection observed at 45 degrees is unrelated.",
                ],
            )
        ]
    )

    _name_ids_in_prose(record)

    comparison = record.tournament.comparisons[0]
    printed = " ".join([*comparison.debate_turns, comparison.rationale])
    assert "methods_statistics" not in printed
    assert "ethics_safety_governance" not in printed
    assert "methods_feasibility" not in printed and "safety_governance" not in printed
    assert "the methods and statistics review rightly points out" in printed
    assert "the ethics, safety and governance review and" in printed
    # The person, where the sentence was built around the person.
    assert "the evidence and correctness reviewer both stand behind it" in printed
    assert "The safety review finding is the only one left open." in printed
    assert "The feasibility review is answered by the protocol." in printed
    # A stage id standing as the subject, with no noun after it to give it away.
    assert "The evidence and correctness review also noted the risk" in printed
    # And the word in its own right, which a report on materials is entitled to use.
    assert "The specular reflection observed at 45 degrees is unrelated." in printed


def test_every_reviewer_the_report_names_is_a_reviewer_a_panelist_can_name():
    """The substitution above holds its own table of ids, and a specialist added to
    the roster without an entry in it goes back to reaching the reader as an enum."""
    from coscientist.narrative import _REVIEW_IDS, _REVIEWER_NAMES

    for reviewer, name in _REVIEWER_NAMES.items():
        assert reviewer in _REVIEW_IDS, reviewer
        # The table is keyed on the pass; the roster names the person doing it.
        assert _REVIEW_IDS[reviewer].replace("review", "reviewer") == name.lower()


def _discovery_with(statements: list[str], leads: list[SourceLead]):
    """A record whose discovery recorded those statements under one direction."""
    from coscientist.models import DiscoveryManifest, DiscoveryNarrative
    from coscientist.narrative import ResearchRecord

    session = Session(question="Does a coating help?")
    record = ResearchRecord(session=session)
    record.discovery = DiscoveryManifest(
        question="Does a coating help?",
        source_leads=leads,
        narratives=[
            DiscoveryNarrative(
                question="Does a coating help?",
                research_directions=["Coating thickness against retention."],
                statements=[
                    DiscoveryStatement(
                        text=text, facet="supporting", originating_pass=1
                    )
                    for text in statements
                ],
            )
        ],
    )
    return record


def test_a_table_row_and_a_bare_source_name_are_not_printed_as_findings():
    """A live Key Findings list held both, under a lead-in calling them findings.

    One entry was a markdown table printed a row at a time, header pipes and all.
    Nine more were a source's own name -- "NextGenBat aalto.fi" -- which says
    nothing about the field and is the reference-list entry verbatim.
    """
    from coscientist.narrative import (
        _evidence_statements,
        _section_three,
        _unstated_findings,
    )

    record = _discovery_with(
        [
            "A 2 nm alumina coating held retention past five hundred cycles.",
            "| Evaluated Literature | Coating Thickness | Electrolyte |",
            "NextGenBat aalto.fi",
            "Applications of Atomic Layer Deposition - MDPI mdpi.com",
        ],
        [
            SourceLead(canonical_url="https://aalto.fi/x", title="NextGenBat aalto.fi"),
            SourceLead(
                canonical_url="https://mdpi.com/y",
                title="Applications of Atomic Layer Deposition - MDPI mdpi.com",
            ),
        ],
    )

    printed = [statement.text for statement in _evidence_statements(record)]

    assert printed == [
        "A 2 nm alumina coating held retention past five hundred cycles."
    ]
    assert _unstated_findings(record) == 3
    prose = " ".join(_section_three(record).core)
    assert "is one finding from the literature" in prose
    # No silent caps: the three that were dropped are counted where the one is.
    assert "Three further statements were recorded and are not printed below" in prose
    assert "a source's own name, or a row of a table" in prose


def test_a_findings_list_that_drops_nothing_says_nothing_about_dropping():
    """The admission is a report of a defect, not a standing disclaimer."""
    from coscientist.narrative import _section_three, _unstated_findings

    record = _discovery_with(
        [
            "A 2 nm alumina coating held retention past five hundred cycles.",
            "Thicker coatings raised the impedance.",
        ],
        [SourceLead(canonical_url="https://aalto.fi/x", title="NextGenBat aalto.fi")],
    )

    assert _unstated_findings(record) == 0
    prose = " ".join(_section_three(record).core)
    assert "two findings from the literature" in prose
    assert "not printed below" not in prose


def test_a_findings_list_counts_how_many_of_them_name_a_source():
    """The lead-in said every finding was "set out below with the source each came
    from", over a list in which eleven of thirty-three printed no source at all.

    Discovery records a finding without a URL and the finding then prints bare --
    "Experimental solid layers often deviate massively due to defects", "One primary
    study testing tungsten oxide coatings on CR2032 cells conducted its evaluations
    at 20°C and 40°C", nine more. Under a sentence promising all of them were
    attributed, a missing number reads as a typesetting slip rather than as the one
    thing worth knowing about that finding: nothing on the record checks it.
    """
    from coscientist.models import DiscoveryManifest, DiscoveryNarrative
    from coscientist.narrative import CitationRegistry, ResearchRecord, _section_three

    lead = SourceLead(canonical_url="https://aalto.fi/x", title="Coatings and cycling")
    session = Session(question="Does a coating help?")
    record = ResearchRecord(session=session)
    record.citations = CitationRegistry([lead])
    record.discovery = DiscoveryManifest(
        question="Does a coating help?",
        source_leads=[lead],
        narratives=[
            DiscoveryNarrative(
                question="Does a coating help?",
                research_directions=["Coating thickness against retention."],
                statements=[
                    DiscoveryStatement(
                        text="A 2 nm alumina coating held retention past 500 cycles.",
                        facet="supporting",
                        originating_pass=1,
                        source_urls=["https://aalto.fi/x"],
                    ),
                    DiscoveryStatement(
                        text="Experimental solid layers often deviate due to defects.",
                        facet="supporting",
                        originating_pass=1,
                    ),
                    DiscoveryStatement(
                        text="One study cycled its CR2032 cells at 20°C and 40°C.",
                        facet="supporting",
                        originating_pass=1,
                    ),
                ],
            )
        ],
    )

    prose = " ".join(_section_three(record).core)

    assert "three findings from the literature, set out below." in prose
    assert "One finding carries the source it came from." in prose
    assert (
        "Discovery recorded none against the other two findings, which print bare "
        "below, and there is nothing on the record to check them against." in prose
    )
    # The promise the list could not keep.
    assert "with the source each came from" not in prose


def test_a_findings_list_where_every_finding_names_a_source_says_so():
    """The admission is a report of a defect, not a standing disclaimer: a fully
    attributed list should not be told two thirds of the sentence about bare ones."""
    from coscientist.models import DiscoveryManifest, DiscoveryNarrative
    from coscientist.narrative import CitationRegistry, ResearchRecord, _section_three

    lead = SourceLead(canonical_url="https://aalto.fi/x", title="Coatings and cycling")
    session = Session(question="Does a coating help?")
    record = ResearchRecord(session=session)
    record.citations = CitationRegistry([lead])
    record.discovery = DiscoveryManifest(
        question="Does a coating help?",
        source_leads=[lead],
        narratives=[
            DiscoveryNarrative(
                question="Does a coating help?",
                research_directions=["Coating thickness against retention."],
                statements=[
                    DiscoveryStatement(
                        text="A 2 nm alumina coating held retention past 500 cycles.",
                        facet="supporting",
                        originating_pass=1,
                        source_urls=["https://aalto.fi/x"],
                    )
                ],
            )
        ],
    )

    prose = " ".join(_section_three(record).core)

    assert "Every one of them carries the source it came from." in prose
    assert "print bare" not in prose


def test_a_source_name_is_matched_however_the_reference_list_trims_it():
    """The statement copies the raw title; the reference list cuts the hostname."""
    from coscientist.narrative import _folded_title, _recorded_titles, _states_a_finding

    record = _discovery_with(
        [],
        [
            SourceLead(
                canonical_url="https://mdpi.com/y",
                title="Atomic Layer Deposition - MDPI mdpi.com",
            )
        ],
    )
    titles = _recorded_titles(record)

    assert not _states_a_finding("Atomic Layer Deposition - MDPI mdpi.com.", titles)
    assert not _states_a_finding("Atomic Layer Deposition - MDPI", titles)
    assert _states_a_finding("Atomic layer deposition raised the impedance.", titles)
    assert _folded_title(" A  Title. ") == "a title"


def test_a_draft_a_researcher_edited_is_not_counted_as_a_stage_gate(
    rich_session: Session,
):
    """Every decision was counted, so a live run reported "nine gate decisions" two
    lines under "Stages completed: 8 of 8". The ninth was a researcher's edit to the
    scope draft, which opens a gate rather than closing one."""
    rich_session.decisions.append(
        HumanDecision(
            action=DecisionAction.REVISE,
            stage="scope",
            actor="web_researcher",
            automatic=False,
            session_version=1,
        )
    )
    for stage, automatic in (
        ("scope", False),
        ("evidence", True),
        ("generate", True),
        ("rank", False),
    ):
        rich_session.decisions.append(
            HumanDecision(
                action=DecisionAction.ACCEPT,
                stage=stage,
                actor="milestone_auto_policy" if automatic else "web_researcher",
                automatic=automatic,
                session_version=1,
            )
        )

    facts = _run_block(compile_dossier(rich_session))
    approvals = next(line for line in facts if line.startswith("- Approvals:"))

    assert "four stage gates" in approvals
    assert "two of this run's four stage gates were accepted automatically" in approvals
    # Not "the rest": the second count is stated so the arithmetic can be checked.
    assert "the other two by a person" in approvals
    assert "gate decision" not in approvals
    # The edit is a fact about the run, so it is reported rather than dropped.
    assert "one draft was sent back for revision before being accepted" in approvals
    assert "(scoping the goal)" in approvals


def test_a_claim_recorded_where_a_title_belongs_is_not_printed_as_a_paper_name():
    """Entry 11 of a live reference list read "The most critical failure point in the
    available scientific literature is the sample size requirement." -- the finding,
    printed a second time as the name of the paper it was drawn from. Entry 20 was a
    forty-word sentence about electrolyte formulations. Both came in from the evidence
    stage, which records a statement and its source side by side."""
    from coscientist.narrative import _reference_title

    claim = SourceLead(
        canonical_url="https://chemrxiv.org/doi/10.26434/chemrxiv.15001487",
        title=(
            "The most critical failure point in the available scientific literature "
            "is the sample size requirement"
        ),
    )
    assert _reference_title(claim) == "Untitled source on chemrxiv.org"

    # A long title in the case a journal sets one in is a title, and stays.
    for kept in (
        "Advances in Coating Materials for Silicon-Based Lithium-Ion Battery Anodes",
        "Recent advances in lithium metal protective strategies with a stable interface",
        "Bulk properties and transport mechanisms of a solid state antiperovskite "
        "Li-ion conductor Li3OCl: insights from first principles calculations",
    ):
        assert _reference_title(
            SourceLead(canonical_url="https://x/1", title=kept)
        ) == (kept)


def test_a_claim_is_caught_by_what_it_carries_not_by_the_verb_it_reached_for():
    """The guard above tests for a clause verb off a list tuned against the sixty-
    three titles of one run. The next run's statements reached for confirm, verifies
    and requires, none of them on it, and three of twenty reference entries were
    claims printed as the names of papers -- one of them forty-five words ending
    "[safety_governance; https://www." in a bibliography. Two signals decide it
    without guessing at wording: markup no title carries, and an opener none uses."""
    from coscientist.narrative import _reference_title

    for statement in (
        "Experimental protocols strictly adhering to the use of standardized CR2032 "
        "coin cells equipped with a defined lithium metal foil counter electrode "
        "confirm that uncoated baseline cells suffer from severe dendrite "
        "proliferation [safety_governance; https://www.",
        "While primary supporting literature heavily verifies the mechanistic safety "
        "benefits of ALD coatings, comprehensive scientific due diligence requires "
        "addressing all evidence types [safety_governance; https://www",
        "Furthermore, explicit compliance with international safety and governance "
        "testing frameworks—most notably UN 38",
    ):
        assert (
            _reference_title(
                SourceLead(
                    canonical_url="https://www.mdpi.com/2075-4701/15/8/892",
                    title=statement,
                )
            )
            == "Untitled source on mdpi.com"
        ), statement[:40]

    # The widened verb list still has to leave a paper named with one of those verbs
    # alone. Title case is what separates them, and it is checked before the verb.
    for kept in (
        "Operando XRD Reveals Phase Transitions in Ni-Rich NMC811 Cathodes During "
        "Extended High-Voltage Cycling",
        "In Situ Spectroscopy Shows That Alucone Coatings Suppress Transition Metal "
        "Dissolution in Layered Oxides",
    ):
        assert (
            _reference_title(SourceLead(canonical_url="https://x/1", title=kept))
            == kept
        )


def test_an_entry_that_has_only_the_authors_says_it_has_no_title():
    """ "22. Zhao et al." was a whole reference entry: the authors came out of a table
    cell in a pass report and the title stayed in the next column."""
    from coscientist.narrative import _reference_title

    lead = SourceLead(
        canonical_url="https://pubs.acs.org/aamick/article/16/10/13029/88607",
        title="| Zhao et al",
    )

    assert _reference_title(lead) == ("Zhao et al., untitled source on pubs.acs.org")


def test_the_minority_lead_in_counts_the_entries_not_the_cases_it_distinguishes():
    """ "The remaining entries are the inverse case, a protected minority: one where a
    region rests on a single idea" stood over two bullets on a live run. "one where"
    opens a list of the cases the run tells apart, and a reader counting what the
    sentence promises finds one entry against the two below it."""
    from coscientist.narrative import _ConnectionCounts, connections_lead_in

    two_alike = connections_lead_in(_ConnectionCounts(converging=1, sole_minority=2))
    assert "The remaining two entries are the inverse case" in two_alike
    assert "one where a region rests" not in two_alike
    assert "in each, a region rests on a single idea" in two_alike
    assert "They are about how thinly a region is covered" in two_alike

    alone = connections_lead_in(_ConnectionCounts(converging=1, sole_minority=1))
    assert "The remaining entry is the inverse case" in alone
    assert "minority: a region rests on a single idea" in alone
    assert "It is about how thinly a region is covered" in alone

    # Two cases really present is the one place a list of cases belongs, and then
    # each is counted so the two counts add up to the entries below.
    both = connections_lead_in(
        _ConnectionCounts(converging=1, sole_minority=2, shared_minority=1)
    )
    assert "The remaining three entries are the inverse case" in both
    assert "two entries where a region rests on a single idea" in both
    assert "one entry where a region has more than one occupant" in both


def test_the_minority_lead_in_stops_saying_remaining_when_it_is_not_last():
    """ "The remaining two entries are the inverse case" stood over two minority
    bullets and a third bullet under them, on a live run: an entry whose ids reach no
    idea is printed after the ones that can be named, so "remaining" -- which is a
    claim about what is left in the list, not a count of a kind -- was false by one.
    """
    from coscientist.narrative import _ConnectionCounts, connections_lead_in

    trailed = connections_lead_in(
        _ConnectionCounts(converging=4, shared_minority=2, unnameable=1)
    )
    assert "Two further entries are the inverse case" in trailed
    assert "remaining" not in trailed

    one_trailed = connections_lead_in(
        _ConnectionCounts(converging=4, sole_minority=1, unnameable=2)
    )
    assert "One further entry is the inverse case" in one_trailed
    assert "It is about how thinly a region is covered" in one_trailed
    assert "remaining" not in one_trailed

    # Nothing below the minority notes and "remaining" is the true word for them.
    assert "The remaining two entries" in connections_lead_in(
        _ConnectionCounts(converging=4, shared_minority=2)
    )


def test_one_finding_stated_at_three_lengths_is_one_finding():
    """A live run printed the same finding about electrolyte standardisation three
    times, in three different relation groups: once as the pass's own section heading
    plus its first sentence, once as that sentence with the rest of the paragraph, and
    once with both. The section then told the reader discovery had read each of them
    differently, which is a fact about the merge and not about the literature."""
    from coscientist.narrative import _EvidenceStatement, _merged_statements

    heading = "B. Electrolyte Standardization (1M LiPF6 in EC:EMC 3:7 wt% + 2 wt% VC) "
    opening = (
        "Several studies within the current battery materials research utilize the "
        "specified Gen2 baseline electrolyte formulation, consisting of 1M LiPF6 in "
        "an ethylene carbonate (EC) and ethyl methyl carbonate (EMC) blend at a 3:7 "
        "weight ratio, combined with 2 wt% vinylene carbonate (VC) additives."
    )
    rest = (
        " However, none of the studies employing this exact standardized formulation "
        "applied a 2-5 nm protective coating to evaluate capacity retention against "
        "an uncoated control."
    )
    copies = [
        (heading + opening, "supports", "https://x/1"),
        (opening + rest, "contradicts", "https://x/2"),
        (heading + opening + rest, "neutral", "https://x/3"),
    ]

    merged = _merged_statements(
        [
            _EvidenceStatement(text=text, urls=[url], facet="claim", relation=relation)
            for text, relation, url in copies
        ]
    )

    assert len(merged) == 1
    # The fullest statement of it is the one printed, and no locator is lost.
    assert merged[0].text == heading + opening + rest
    assert merged[0].urls == ["https://x/3", "https://x/1", "https://x/2"]
    assert merged[0].relation == "recorded_both_ways"


def test_a_finding_and_the_same_finding_with_the_pass_s_own_aside_are_one():
    """One pass wrote where it had read the thing into the sentence saying what it
    found -- "(e.g., TUM investigations, https://mediatum.ub.tum.de/...)" -- and
    another did not, so an exact match saw two findings and printed them apart."""
    from coscientist.narrative import _EvidenceStatement, _merged_statements

    bare = (
        "Although 25 C is occasionally cited as a standard testing temperature for "
        "general battery cycling, it is never paired with the n=5 sample size "
        "constraint."
    )
    cited = (
        "Although 25 C is occasionally cited as a standard testing temperature for "
        "general battery cycling (e.g., TUM (Technical University of Munich) "
        "investigations, https://mediatum.ub.tum.de/doc/1691934/1691934.pdf), it is "
        "never paired with the n=5 sample size constraint."
    )

    merged = _merged_statements(
        [
            _EvidenceStatement(text=bare, urls=[], facet="claim", relation="neutral"),
            _EvidenceStatement(text=cited, urls=[], facet="claim", relation="neutral"),
        ]
    )

    assert len(merged) == 1
    # The aside is dropped from the comparison, not from the page.
    assert merged[0].text == cited


def test_a_short_finding_inside_a_longer_one_is_left_alone():
    """Containment is how a paragraph swallows its own first sentence. A clause short
    enough to turn up inside an unrelated paragraph is not the same finding."""
    from coscientist.narrative import _EvidenceStatement, _merged_statements

    short = "Coatings raise impedance."
    long = (
        "Across the retrieved literature the consensus is unsettled, and one review "
        "notes in passing that coatings raise impedance. under some deposition "
        "conditions that were not otherwise characterised."
    )

    merged = _merged_statements(
        [
            _EvidenceStatement(text=short, urls=[], facet="claim", relation="supports"),
            _EvidenceStatement(text=long, urls=[], facet="claim", relation="neutral"),
        ]
    )

    assert len(merged) == 2


def test_a_section_heading_flattened_onto_a_finding_is_punctuated_not_run_on():
    """The pass writes Markdown and records findings cut from it. Where the cut began
    at a heading the hashes went and the words did not, so a live Key Findings list
    opened "B. Electrolyte Standardization (...) Several studies within" -- two
    sentences printed as one, reading as neither."""
    from coscientist.narrative import _with_heading_separated

    assert _with_heading_separated(
        "B. Electrolyte Standardization (1M LiPF6 in EC:EMC 3:7 wt% + 2 wt% VC) "
        "Several studies utilize the specified baseline."
    ) == (
        "Electrolyte Standardization (1M LiPF6 in EC:EMC 3:7 wt% + 2 wt% VC): "
        "Several studies utilize the specified baseline."
    )
    assert _with_heading_separated(
        "A. Sample Size (n = 5) The most critical failure point is the sample size."
    ) == ("Sample Size (n = 5): The most critical failure point is the sample size.")


def test_a_finding_that_is_not_a_flattened_heading_is_left_as_it_was():
    """The boundary is only findable where the heading ends on a bracket. Anything
    else is left alone rather than cut at a guess."""
    from coscientist.narrative import _with_heading_separated

    intact = [
        "A 3 nm coating (deposited by ALD) improves retention at 500 cycles.",
        "C. Coating Thickness Several studies report a 2 nm floor.",
        "The coating (2 to 5 nm) raises the impedance of the cell.",
    ]

    for text in intact:
        assert _with_heading_separated(text) == text


def _briefed(session: Session, briefing: str, author: str) -> str:
    for artifact in session.artifacts:
        if artifact.schema_name == "TournamentState":
            artifact.payload["briefing"] = briefing
            artifact.payload["briefing_author"] = author
    return compile_dossier(session)


JUDGE_BRIEFING = (
    "The coating idea separated from the field on falsifiability rather than on "
    "impact: it is the only finalist whose failure condition the reviews can "
    "check without new equipment. Second and third are eleven points apart, a "
    "third of one match, so that order should be treated as unsettled."
)


def test_the_judges_reading_of_the_tournament_sits_under_the_ranked_table(
    rich_session: Session,
):
    """A column of Elo says which idea finished ahead, not what decided it.

    The reader of the summary table is choosing between eight ideas on one page,
    and the gaps in that column are the part they cannot interpret alone: the
    judge that played the matches says which of them are too narrow to read
    anything into.
    """
    report = _briefed(rich_session, JUDGE_BRIEFING, "judge")

    summary = report.split(SUMMARY_TABLE_HEADING)[1].split("\n## ")[0]
    assert "**What the tournament found.**" in summary
    assert JUDGE_BRIEFING in summary


def test_the_computed_fallback_does_not_restate_the_table_it_sits_under(
    rich_session: Session,
):
    """Arithmetic over the match record is true and is not a briefing.

    It is the standings table in sentences, printed directly below the standings
    table, and it would carry the authority of a judge's reading while saying
    nothing the row above it does not.
    """
    computed = "Field: 6 hypotheses, 6 matches (6 decided, 0 drawn).\nFinal standings:"

    report = _briefed(rich_session, computed, "computed")

    assert "**What the tournament found.**" not in report
    assert "Final standings:" not in report


def test_a_claim_cannot_stand_better_than_the_paper_it_was_drawn_from():
    """A claim carries no record of whether its document came back, and the citation
    read only the claim. So a claim extracted before the run went back to its source
    and failed to retrieve it stayed "discovered_unverified": the reference entry for
    that paper read "could not be retrieved ... nothing here is grounded by it", the
    bullet citing it was badged a literature lead for a reader to go and follow, and
    the idea resting on it was reported unverified rather than discredited."""
    from coscientist.citations import resolve_candidate
    from coscientist.models import (
        Candidate,
        EvidenceClaim,
        EvidencePacket,
        SourceRecord,
    )

    packet = EvidencePacket(
        question="Does a coating help?",
        sources=[
            SourceRecord(
                id="src",
                url="https://example.org/thermal",
                title="Operando gas analysis of NMC811",
                verification_status="inaccessible",
            )
        ],
        claims=[
            EvidenceClaim(
                id="claim",
                claim="Uncoated NMC811 releases oxygen from 140 °C.",
                source_id="src",
                verification_status="discovered_unverified",
            )
        ],
    )
    candidate = Candidate(
        title="Coat it",
        claim="A coating raises the onset temperature.",
        rationale="The oxygen release is the failure mode.",
        mechanism_model="Scavenging",
        validation_protocol="DSC-MS",
        falsifier="No shift in the exotherm",
        evidence_ids=["claim"],
    )

    resolved = resolve_candidate(candidate, packet)

    assert resolved.support == "discredited"
    assert resolved.discrediting_statuses == frozenset({"inaccessible"})
    # And the citation says which of the two verdicts, so the warning naming it can.
    assert [item.status for item in resolved.citations] == ["inaccessible"]


def test_an_evidence_gap_that_names_a_broken_record_is_badged_as_one(
    rich_session: Session,
):
    """The record that discredited the top-ranked idea of a live run was cited in an
    evidence gap and nowhere else. Gaps carry no badge -- a gap is a statement that no
    evidence exists -- so the chapter opened on "its stated grounding is discredited"
    over bullets that all read verified or followable, and nothing on the page was the
    broken citation the warning is about."""
    from coscientist.narrative import DISCREDITED_BADGE, _evidence_notes

    record = load_record(rich_session)
    broken = next(
        claim
        for claim in record.evidence.claims
        if claim.verification_status == "inaccessible"
    )
    candidate = record.population.candidates[0]
    record.cited_evidence[candidate.id] = [[], [], [broken.id]]

    notes = _evidence_notes(record, candidate)
    gaps = [
        (badge, text) for heading, badge, text in notes if heading == "Evidence gaps"
    ]

    assert gaps and gaps[0][0] == DISCREDITED_BADGE
    # The statement itself is still the record's, not the id the specialist wrote.
    assert broken.claim.rstrip(".") in gaps[0][1]


def test_a_gap_that_names_nothing_still_carries_no_badge(rich_session: Session):
    """Grounding is not a question that can be asked of "no study has measured this",
    and a badge there would offer a reader a document to go and look for."""
    from coscientist.narrative import _evidence_notes

    record = load_record(rich_session)
    candidate = record.population.candidates[0]
    record.cited_evidence[candidate.id] = [
        [],
        [],
        ["No published study cycles these cells beyond 500 cycles."],
    ]

    notes = _evidence_notes(record, candidate)

    assert [badge for heading, badge, _ in notes if heading == "Evidence gaps"] == [""]


def _generated_by(rich_session: Session, agent: str) -> str:
    """The provenance appendix of a run whose generation stage fanned out."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import ProvenanceNote

    record = load_record(rich_session)
    record.provenance.append(
        ProvenanceNote(
            stage="generate",
            agent=agent,
            schema_name="CandidatePopulation",
            source="repaired",
            repairs=["Candidate.score_novelty 8 -> 4 (answered on a 1-10 scale)"],
            error="",
            model="gemini-3.1-pro-preview",
        )
    )
    return "\n".join(_provenance_appendix(record))


def test_the_four_generators_are_named_as_the_summary_table_names_them(
    rich_session: Session,
):
    """Missing from the name table, a generator fell through to its id with the
    underscores taken out. The appendix credited "generation evidence first" for work
    the Executive Candidate Summary files under "evidence first", and three of the
    four repair paragraphs -- the only record of what the run changed -- opened on a
    pipeline id as their grammatical subject."""
    appendix = _generated_by(rich_session, "generation_evidence_first")

    assert "generation evidence first" not in appendix
    assert "| idea generation | evidence-first generation |" in appendix
    assert "The evidence-first generation answer was repaired" in appendix


def test_the_merge_of_the_generators_is_named_and_not_filed(rich_session: Session):
    appendix = _generated_by(rich_session, "generation_aggregator")

    assert "generation aggregator" not in appendix
    assert "The idea aggregation answer was repaired" in appendix


def test_the_written_by_column_answers_the_question_its_header_asks(
    rich_session: Session,
):
    """The column printed the enum the payload is filed under, so four rows of a live
    table answered "Written by" with "repaired" -- which names neither an author nor
    anything the page has defined by the time a reader meets it."""
    appendix = _generated_by(rich_session, "generation_evidence_first")
    table = appendix.split("## What each stage produced")[1]

    assert "| Written by |" in table
    assert "| the specialist, then repaired |" in table
    assert "| repaired |" not in table


def test_the_repair_caveat_is_printed_once_under_all_the_stages_it_covers(
    rich_session: Session,
):
    """Four stages were repaired the same way, so the same thirty-one words closed
    four consecutive paragraphs -- the shape the grouping exists to take out of one
    sentence, put back around four of them."""
    from coscientist.dossier import _provenance_appendix
    from coscientist.narrative import ProvenanceNote

    record = load_record(rich_session)
    for agent in ("generation_evidence_first", "generation_mechanism_first"):
        record.provenance.append(
            ProvenanceNote(
                stage="generate",
                agent=agent,
                schema_name="CandidatePopulation",
                source="repaired",
                repairs=[
                    f"Candidate.score_novelty {raw} -> {raw // 2} "
                    "(answered on a 1-10 scale)"
                    for raw in (8, 6)
                ],
                error="",
                model="gemini-3.1-pro-preview",
            )
        )
    appendix = "\n".join(_provenance_appendix(record))

    assert appendix.count("in the order it met them") == 1
    # And it still stands under both of the paragraphs it is about.
    assert appendix.index("mechanism-first generation") < appendix.index(
        "in the order it met them"
    )


def test_a_fork_counts_both_the_stages_it_started_past(rich_session: Session):
    """ "8 of 8" less the one stage the bullet disclaimed read as seven stages of this
    run's own work. A fork carries the scope over too, so it was six."""
    rich_session.seeded_evidence_from = "session_earlier"

    block = compile_dossier(rich_session).split("\n## Run\n", 1)[1]

    assert "includes the scope and evidence stages it started past" in block


def test_a_fork_says_which_of_the_models_it_names_it_never_called(
    rich_session: Session,
):
    """ "Produced by" is the field an auditor reproduces a run from. On a fork it
    listed the Deep Research model that built the forked corpus, unqualified, one
    bullet under the line saying the run did not search the literature."""
    from coscientist.narrative import ProvenanceNote

    record = load_record(rich_session)
    record.session.seeded_evidence_from = "session_earlier"
    record.provenance.append(
        ProvenanceNote(
            stage="evidence",
            agent="deep_research_discovery",
            schema_name="DiscoveryManifest",
            source="specialist",
            repairs=[],
            error="",
            model="deep-research-preview-04-2026",
        )
    )
    from coscientist.dossier import _run_facts

    produced = next(
        fact for fact in _run_facts(record) if fact.startswith("- Produced by:")
    )

    assert "deep-research-preview-04-2026" in produced
    assert (
        "of which deep-research-preview-04-2026 produced the forked scope and "
        "evidence rather than anything this run ran" in produced
    )


def test_the_warnings_chapter_says_the_evidence_base_is_not_this_runs(
    rich_session: Session,
):
    """The overview points at Warnings and Limitations as the place where every
    limitation on the report is set out. A fork's chapter said nothing about the one
    structural fact -- that the corpus was never searched here -- so a reader who did
    what the overview told them to learned it only from Provenance, thirty pages on,
    if at all."""
    rich_session.seeded_evidence_from = "session_earlier"

    report = compile_dossier(rich_session)
    chapter = report.split("# Warnings and Limitations")[1].split("\n# ")[0]

    assert "## An evidence base this run did not gather" in chapter
    assert "session_earlier" in chapter
    # And the standing limits no longer count a search among the stages this run ran.
    assert "Every stage of it is desk work: a literature search" not in chapter
    assert (
        "desk work: a set of proposals written by models over literature an "
        "earlier run gathered" in chapter
    )


def test_a_run_that_searched_for_itself_keeps_the_search_in_its_desk_work(
    rich_session: Session,
):
    report = compile_dossier(rich_session)

    assert "An evidence base this run did not gather" not in report
    assert "Every stage of it is desk work: a literature search" in report


def _chapter_lead(rich_session: Session, opening: str) -> str:
    """The paragraph of the compiled report that starts with ``opening``."""
    return next(
        line
        for line in compile_dossier(rich_session).splitlines()
        if line.startswith(opening)
    )


def test_the_generation_strategies_are_named_where_the_chapter_counts_them(
    rich_session: Session,
):
    """The chapter opened "across the strategies described above" over an idea count.
    Nothing above it holds such a list -- the chapters before describe research
    directions and mechanism clusters, and neither uses the word -- so a reader
    looking back for the set to read eight ideas against found no set."""
    lead = _chapter_lead(rich_session, "The generator produced")

    assert "described above" not in lead
    assert (
        "Four generation strategies worked the question in parallel: evidence "
        "first, mechanism first, analogy transfer, and competing explanation." in lead
    )


def test_a_run_that_recorded_no_strategy_names_none(rich_session: Session):
    """The names come off the candidates, so a run that recorded none has to open the
    chapter without a dangling colon where the series would have been."""
    from dataclasses import replace

    from coscientist.narrative import _section_four, build_idea_briefs

    record = load_record(rich_session)
    briefs = [replace(brief, strategy="") for brief in build_idea_briefs(record)]

    lead = _section_four(record, briefs).core[0]

    assert "worked the question in parallel" not in lead
    assert lead.startswith(
        "The generator produced six ideas, each stated as a claim that can be "
        "shown to be wrong. They are set out here in rank order"
    )


def test_the_review_passes_are_named_as_the_rest_of_the_report_names_them(
    rich_session: Session,
):
    """Written out by hand, the lead-in invented two names nothing else uses. A reader
    who went looking for the "methods and feasibility" or "safety and governance"
    pass found the methods and statistics reviewer and the ethics, safety and
    governance reviewer on the cover, and no way to tell whether five had run or
    seven."""
    lead = _chapter_lead(rich_session, "Five independent reviews were run")

    assert (
        "Five independent reviews were run against each idea: evidence and "
        "correctness; novelty; methods and statistics; impact; and ethics, safety "
        "and governance." in lead
    )
    assert "methods and feasibility" not in lead
    assert "reviews were run against each idea: safety and governance" not in lead


def test_the_score_legend_counts_the_routes_to_a_two_that_it_lists(
    rich_session: Session,
):
    """ "Two different reviews therefore print the same number" opened a colon that
    then listed three ways of reaching a two, so a reader counting them found one
    more than the sentence promised -- in the paragraph that defines how every score
    in the report is to be read."""
    lead = _chapter_lead(rich_session, "Five independent reviews were run")

    assert "Reviews that reached different verdicts therefore print the same" in lead
    assert "Two different reviews therefore print" not in lead


def test_the_score_legend_sends_the_reader_where_both_halves_are_printed(
    rich_session: Session,
):
    """It sent them to Deep Verification, which holds the fatal flaws and the
    objections -- the one subsection that prints neither the recommendation nor the
    confidence the sentence has just said the number cannot be read without."""
    lead = _chapter_lead(rich_session, "Five independent reviews were run")
    pointer = lead.split("has to be read off the review itself,")[1]

    assert "printed under Reviews in the idea's own section" in pointer
    assert "Deep Verification" not in pointer


def test_the_success_criteria_are_set_apart_only_where_there_is_a_second_block(
    rich_session: Session,
):
    """The cover prints the comparison criteria only where the run recorded some, so
    on a run that recorded none the sentence sent a reader back to page one for a
    block that is not there -- four lines after the same section had told them no
    cross-candidate criterion was recorded."""
    from coscientist.narrative import _section_two

    record = load_record(rich_session)
    record.population.comparison_criteria = []

    lead = next(
        paragraph
        for paragraph in _section_two(record).core
        if paragraph.startswith("Success for the goal as a whole")
    )

    assert "is stated under Criteria on the cover. No stage of this run" in lead
    assert "apart from the comparison criteria" not in lead
    assert "score well under the reviews and still fail to advance the goal" in lead


def test_a_goal_too_long_to_print_whole_is_still_shortened_in_its_own_mood(
    rich_session: Session,
):
    """Deriving a title strips the terminal punctuation, so a question of more than
    twelve words came back as a statement and the cover asserted what the goal,
    printed under it, asks."""
    rich_session.question = (
        "Can a thin protective surface coating meaningfully improve the cycle life "
        "of commercial lithium-ion battery cathodes under fast charging?"
    )

    report = compile_dossier(rich_session)

    assert report.splitlines()[0].endswith("?")
    # And the chapter headings that hang the title off a preposition stay out of it.
    assert "Ranked Research Ideas for Can a" not in report
