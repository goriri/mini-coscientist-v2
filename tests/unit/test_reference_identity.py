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


def test_the_site_spelled_out_goes_the_way_its_hostname_does():
    """Six of the seventeen entries of a live reference list ended in the name of the
    site the search found them on -- "Transition metal dissolution from Li-ion battery
    cathodes - Atomic Layer Deposition", "... in aqueous zinc battery research - The
    Royal Society of Chemistry", "Unexpected high power performance of atomic layer
    deposition coated Li[Ni1/3Mn1/3Co1/3]O2 cathodes - University of Colorado Boulder"
    -- each reading as the last words of the paper's name. The hostname cut had
    already gone through them and left the site written out in front of it.

    The locator is what proves it furniture: spelled out, in one of its words, or as
    the initials the site is registered under."""
    spelled = _lead(
        "https://www.atomiclayerdeposition.com/storage/app/1224/x.pdf",
        title=f"{PAPER} - Atomic Layer Deposition",
    )
    worded = _lead(
        "https://www.colorado.edu/lab/georgegroup/files/376.pdf",
        title=f"{PAPER} - University of Colorado Boulder",
    )
    initialled = _lead(
        "https://pubs.rsc.org/eb/article/1/4/813/891012/",
        title=f"{PAPER} - The Royal Society of Chemistry",
    )

    assert _reference_title(spelled) == PAPER
    assert _reference_title(worded) == PAPER
    assert _reference_title(initialled) == PAPER


def test_the_journal_and_the_publisher_both_go_where_the_title_carries_both():
    """Entry 8 of that list read "Molecular Layer Deposition of Organic-Inorganic
    Hafnium Oxynitride Hybrid Films for Electrochemical Applications | ACS Applied
    Energy Materials - ACS Publications" -- two segments of furniture, one behind the
    other, and cutting only the last would have left the journal reading as a title."""
    lead = _lead(
        "https://pubs.acs.org/doi/10.1021/acsaem.3c01234",
        title=f"{PAPER} | ACS Applied Energy Materials - ACS Publications",
    )

    assert _reference_title(lead) == PAPER


def test_a_subtitle_the_locator_cannot_convict_is_left_on_the_title():
    """The cut is proved against the address, not guessed from the shape of the
    sentence: a paper's own last segment goes to the reader as the paper wrote it."""
    subtitled = _lead(
        "https://www.nature.com/articles/s41560-023-01234",
        title="Silicon anodes - a practical review of the last decade",
    )
    hyphenated = _lead(
        "https://pubs.acs.org/doi/10.1021/acsami.4c12988",
        title="Degradation Effects in Ni-Rich Cells - Learning from Potential Profiles",
    )

    assert _reference_title(subtitled) == (
        "Silicon anodes - a practical review of the last decade"
    )
    assert _reference_title(hyphenated) == (
        "Degradation Effects in Ni-Rich Cells - Learning from Potential Profiles"
    )


def test_a_page_banner_where_the_title_should_be_is_cut_as_the_banner_it_is():
    """A live claim was drawn from "Constructing Safe and Durable High-Voltage P2
    Layered Cathodes for Sodium Ion Batteries Enabled by Molecular Layer Deposition of
    Alucone - Welcome to Chemical Engineering at the University of Waterloo". The
    banner is too long to be read as a site name and shares no word with the address
    it was served from, so it is recognised by what no paper's title ever says."""
    lead = _lead(
        "http://chemeng.uwaterloo.ca/zchen/publications/documents/adfm.pdf",
        title=f"{PAPER} - Welcome to Chemical Engineering at the University of Waterloo",
    )

    assert _reference_title(lead) == PAPER


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
