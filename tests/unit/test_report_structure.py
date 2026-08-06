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
    _cell,
    _match_summary,
    _verdict_line,
    compile_dossier,
    shared_match_notes,
)
from coscientist.models import (
    ApprovalProfile,
    DecisionAction,
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
        Citation(number=1, title="A Paper", url="https://www.frontiersin.org/")
    )
    assert front_page == (
        "1. A Paper. Retrieved from frontiersin.org; the literature search recorded "
        "no link to the document itself."
    )
    assert _reference_line(Citation(number=2, title="A Paper", url="")).endswith(
        "No link to this source was recorded, so it has to be found by title."
    )
    # "It has to be found by title" over an entry that has no title is advice the
    # entry refutes: what the search returned for these is one of its own redirects,
    # which names neither the document nor a host.
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ"
    assert (
        _reference_line(
            Citation(number=3, title="Untitled source on nih.gov", url=redirect)
        )
        == "3. Untitled source on nih.gov."
    )


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


# ---------------------------------------------------------------- per-idea layout


def test_each_deep_dive_carries_rank_elo_and_a_category_path(body: str):
    deep_dives = body[body.rindex("\n# Top ideas in detail\n") :]
    ranks = re.findall(
        r"^Rank: (\d+)(?:, tied on Elo with .+)?$", deep_dives, re.MULTILINE
    )
    elos = re.findall(r"^Elo: (\d+)$", deep_dives, re.MULTILINE)
    categories = re.findall(r"^Category: (.+)$", deep_dives, re.MULTILINE)
    assert ranks == [str(n) for n in range(1, len(ranks) + 1)]
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
    assert "Two of them are marked unverified." in text
    assert "uncited" not in text
    assert "grounded" not in text

    quiet, none_hoisted = shared_support_notices(["unverified", "uncited"])
    assert (quiet, none_hoisted) == ("", set())

    # A reader must meet a broken grounding under the idea it belongs to, however
    # many ideas carry one.
    alarming, still_none = shared_support_notices(["unsupported", "unsupported"])
    assert (alarming, still_none) == ("", set())


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
    assert "five are marked unverified and three are marked uncited" in pointer
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
    reviewer's own sentence is only readable while it is short."""
    record = load_record(rich_session)
    statement = record.discovery.narratives[0].statements[0]
    body = _findings(rich_session, f"This is answered by {statement.id}")

    _assert_no_record_ids(body)
    assert "an unverified finding from the literature search" in body


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
    assert "The grounding of the following ideas carries a qualification:" in report

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
        "Rank: 4, tied on Elo with another idea and ordered arbitrarily among them"
    )
    assert brief(4, 2).rank_line == (
        # Spelled, as every other small count in prose is.
        "Rank: 4, tied on Elo with two other ideas and ordered arbitrarily among them"
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
