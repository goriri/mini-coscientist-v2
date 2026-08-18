"""A page can be perfectly retrievable and still not be a scientific source.

One live dossier -- the rehearsal question "does the launcher carry the
rehearsal flag through to the report?", which a grounded search could only match
as the words "rehearsal", "flag" and "smoke test" -- printed a reference list
containing a Steam discussion of RimWorld twice, in Portuguese at [12] and in
Chinese at [15]; the r/halo thread "PRO TIP: The Razorback can hold the flag!"
at [16]; a Call of Duty loadout thread at [18]; "* SECRET * WAYS TO USE SNAPSHOT
GRENADES in REBIRTH" at [17]; and a national day flypast rehearsal on YouTube at
[23]. Two of the six were cited in the body of the report.

Nothing upstream was wrong. Each is a real page at a real address that a
retriever can open, so every test the corpus applied -- does the locator name a
document, does the document exist, does its title read like a title -- passed.
"""

from __future__ import annotations

import pytest

from coscientist.evidence import could_be_literature, merge_leads
from coscientist.models import SourceLead
from coscientist.narrative import CitationRegistry

RIMWORLD = "https://steamcommunity.com/app/294100/discussions/0/6274121610030307411/"
HALO = "https://www.reddit.com/r/halo/comments/r09txw/pro_tip_the_razorback_can/"
FLYPAST = "https://www.youtube.com/watch?v=TiauxNMDbsA"
ACS = "https://pubs.acs.org/doi/10.1021/acsami.4c13335"


@pytest.mark.parametrize(
    ("url", "admissible"),
    [
        (RIMWORLD, False),
        (HALO, False),
        (FLYPAST, False),
        ("https://youtu.be/TiauxNMDbsA", False),
        ("https://www.scribd.com/document/994128175/21CSP302L-VI-Semester", False),
        # The host and everything under it: a subdomain is the same publisher.
        ("https://old.reddit.com/r/halo/comments/r09txw/pro_tip/", False),
        # Absent from the list on purpose. A repository holding a named tool is
        # a citable artifact -- one live corpus cites SyntheMol this way -- and
        # a preprint server is where half of this literature lives.
        ("https://github.com/swansonk14/SyntheMol", True),
        ("https://www.biorxiv.org/content/10.1101/2025.03.18.643954v1", True),
        (ACS, True),
        # Nothing to judge is not a pass; the caller's other guards say so.
        ("", False),
    ],
)
def test_which_hosts_a_finding_can_be_taken_from(url: str, admissible: bool):
    assert could_be_literature(url) is admissible


def test_a_host_that_is_only_a_near_miss_is_not_refused():
    """``x.com`` is on the list. A domain ending in those letters is not it, and
    a substring test would have refused the second along with the first."""
    assert could_be_literature("https://www.embox.com/papers/3") is True
    assert could_be_literature("https://x.com/someone/status/1") is False


def test_the_merge_every_lead_passes_through_drops_the_ones_it_cannot_cite():
    """One place rather than the eight that later read the list. A lead that
    survives here is counted in the source total, sent to a verifier and
    numbered in the references, and the run has nothing true to say about any
    of the three."""
    leads = [
        SourceLead(canonical_url=ACS, title="Alumina interphases"),
        SourceLead(canonical_url=HALO, title="PRO TIP: The Razorback can hold"),
        SourceLead(canonical_url=RIMWORLD, title="Diabolus and combat extended"),
    ]

    merged = merge_leads([], leads)

    assert [lead.canonical_url for lead in merged] == [ACS]


def test_a_corpus_already_holding_one_loses_it_on_the_next_merge():
    """A run that merges again -- a revision, a gap pass -- drops what it holds."""
    stored = [
        SourceLead(canonical_url=HALO, title="PRO TIP: The Razorback can hold"),
        SourceLead(canonical_url=ACS, title="Alumina interphases"),
    ]

    merged = merge_leads(stored, [])

    assert [lead.canonical_url for lead in merged] == [ACS]


def test_the_reference_list_refuses_one_the_stored_corpus_still_holds():
    """The merge governs what a run collects from here on. It does nothing for a
    corpus collected before the rule existed, and re-rendering does not re-merge:
    eight live dossiers went on printing a Steam discussion of RimWorld, an
    r/halo thread, a flypast on YouTube and four Scribd uploads after the merge
    was fixed, because a report is computed on demand from what was stored."""
    registry = CitationRegistry(
        [
            SourceLead(canonical_url=ACS, title="Alumina interphases"),
            SourceLead(canonical_url=HALO, title="PRO TIP: The Razorback can hold"),
            SourceLead(canonical_url=RIMWORLD, title="Diabolus and combat extended"),
        ]
    )

    assert registry.marker([ACS]) == "[1]"
    assert [citation.url for citation in registry.references()] == [ACS]
    assert registry.refused_sources == 2


def test_a_body_citing_a_refused_source_prints_no_marker_for_it():
    """Two of the six reached the body of a report as citations. Dropping the
    entry and leaving "[12]" pointing into a list that no longer has a twelfth
    line would be the worse half of a fix."""
    registry = CitationRegistry(
        [SourceLead(canonical_url=HALO, title="PRO TIP: The Razorback can hold")]
    )

    assert registry.number(HALO) is None
    assert registry.citable([HALO]) is False
    assert registry.references() == []


def test_a_refused_lead_is_not_reported_as_a_duplicate_of_something():
    """The fold's sentence says the leads "named a document another lead had
    already named", which is true of a duplicate and false of a Reddit thread.
    The two counts stay apart so the sentence stays true."""
    registry = CitationRegistry(
        [
            SourceLead(canonical_url=ACS, title="Alumina interphases"),
            SourceLead(canonical_url=HALO, title="PRO TIP: The Razorback can hold"),
        ]
    )

    assert registry.folded_duplicates == 0
    assert registry.refused_sources == 1
