"""One paper found three times is one reference, and the entry says what it has.

References 5, 6 and 7 of a live dossier were the same Royal Society of Chemistry
paper. The grounding API mints a fresh opaque redirect token per citation, three
of them could not be followed to a publisher, and the merge keys on the canonical
URL -- so three tokens meant three references, under a title all three shared word
for word. Each then carried "No link to this source was recorded" over a link that
had been recorded, and a distinguishing clause saying the search had returned them
"without a title" printed directly after the title.
"""

from __future__ import annotations

from coscientist.dossier import _reference_line, _reference_lines
from coscientist.evidence import lead_identity, merge_leads
from coscientist.models import SourceLead
from coscientist.narrative import Citation, _reference_title

REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
PAPER = (
    "Enabling a highly reversible conversion reaction in a lithiated nano-SnO2 "
    "film coated with Al2O3 by atomic layer deposition"
)


def _lead(url: str, **fields) -> SourceLead:
    return SourceLead(canonical_url=url, **fields)


def test_three_redirects_that_share_a_title_are_one_reference():
    leads = [
        _lead(f"{REDIRECT}AUZIYQ{token}", title=PAPER, facets=[facet])
        for token, facet in (
            ("a", "supporting"),
            ("b", "replication"),
            ("c", "methods"),
        )
    ]

    merged = merge_leads([], leads)

    assert len(merged) == 1
    # The facets are why the merge cannot simply drop the later copies: they are
    # what says the paper answered three different searches.
    assert merged[0].facets == ["supporting", "replication", "methods"]


def test_two_redirects_with_different_titles_stay_two_references():
    merged = merge_leads(
        [],
        [
            _lead(f"{REDIRECT}a", title=PAPER),
            _lead(f"{REDIRECT}b", title="A different paper entirely"),
        ],
    )

    assert len(merged) == 2


def test_an_untitled_redirect_is_not_merged_into_every_other_untitled_one():
    merged = merge_leads([], [_lead(f"{REDIRECT}a"), _lead(f"{REDIRECT}b")])

    assert len(merged) == 2


def test_the_same_document_reached_two_ways_merges_on_its_doi():
    merged = merge_leads(
        [],
        [
            _lead(
                f"{REDIRECT}a", title=PAPER, identifiers={"doi": "10.1039/C4TA00001A"}
            ),
            _lead(
                "https://pubs.rsc.org/en/content/articlelanding/2014/ta/c4ta00001a",
                title="Enabling a highly reversible conversion reaction",
                identifiers={"doi": "10.1039/c4ta00001a"},
                verification_status="verified",
                verification_note="Retrieved and read.",
            ),
        ],
    )

    assert len(merged) == 1
    # The locator that reaches the paper wins over the token that does not, and
    # the copy that was read settles the status for the document.
    assert merged[0].canonical_url.startswith("https://pubs.rsc.org/")
    assert merged[0].verification_status == "verified"


def test_a_redirect_and_a_publisher_link_are_told_apart_by_identity():
    assert lead_identity(_lead(f"{REDIRECT}a", title=PAPER)) == f"title:{PAPER.lower()}"
    assert lead_identity(_lead("https://example.org/paper")) == (
        "https://example.org/paper"
    )


def test_the_search_engines_site_name_is_not_part_of_the_papers_title():
    lead = _lead(
        f"{REDIRECT}a",
        title=f"{PAPER} - The Royal Society of Chemistry rsc.org",
    )

    assert _reference_title(lead) == f"{PAPER} - The Royal Society of Chemistry"


def test_a_title_that_is_nothing_but_a_hostname_still_says_it_is_untitled():
    assert _reference_title(_lead("https://example.org/a", title="www.mdpi.com")) == (
        "Untitled source on mdpi.com"
    )


def test_an_unfollowable_redirect_says_a_link_was_recorded_and_stopped_working():
    """ "No link to this source was recorded" was printed over a recorded link."""
    line = _reference_line(Citation(number=5, title=PAPER, url=f"{REDIRECT}a"))

    assert "recorded only its own redirect link" in line
    assert "no longer resolves" in line
    assert "No link to this source was recorded" not in line


def test_a_source_with_no_locator_at_all_still_says_so():
    line = _reference_line(Citation(number=5, title=PAPER, url=""))

    assert "No link to this source was recorded" in line


def test_the_distinguishing_clause_does_not_deny_the_title_it_follows():
    lines = _reference_lines(
        [
            Citation(number=5, title=PAPER, url=f"{REDIRECT}a"),
            Citation(number=6, title=PAPER, url=f"{REDIRECT}b"),
        ]
    )

    assert (
        "the first of two separate records the search returned under that title"
        in (lines[0])
    )
    assert "without a title" not in " ".join(lines)


def test_an_untitled_pair_is_still_distinguished_by_publisher():
    lines = _reference_lines(
        [
            Citation(
                number=5, title="Untitled source on nih.gov", url="https://nih.gov"
            ),
            Citation(
                number=7, title="Untitled source on nih.gov", url="https://nih.gov"
            ),
        ]
    )

    assert "under that publisher without a title" in lines[0]
    assert "the second of two" in lines[1]
