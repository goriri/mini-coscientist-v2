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
from coscientist.evidence import lead_identity, merge_leads, stated_identifiers
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


KRAS = (
    "KRAS Secondary Mutations That Confer Acquired Resistance to KRAS G12C "
    "Inhibitors, Sotorasib and Adagrasib"
)
SCRAPE = "https://www.academia.edu/90000574/KRAS_Secondary_Mutations"
RECORD = "https://pubmed.ncbi.nlm.nih.gov/33971321/"


def test_one_paper_held_at_two_addresses_is_one_reference():
    """Entries 39 and 40 of a live reference list were this one paper.

    One was a scrape on academia.edu and the other the PubMed record, under a
    title the two shared word for word. Neither address carries a DOI, so the
    document identity could not see they were the same study and the reader was
    shown it twice under two numbers. A second run printed the same pair.
    """
    merged = merge_leads(
        [],
        [
            _lead(SCRAPE, title=KRAS, facets=["supporting"]),
            _lead(RECORD, title=KRAS, facets=["replication"]),
        ],
    )

    assert len(merged) == 1
    assert merged[0].facets == ["supporting", "replication"]


def test_the_reference_keeps_whichever_address_is_the_record():
    """A merge that kept the scrape would cite worse than either duplicate did."""
    scrape_first = merge_leads(
        [], [_lead(SCRAPE, title=KRAS), _lead(RECORD, title=KRAS)]
    )
    record_first = merge_leads(
        [], [_lead(RECORD, title=KRAS), _lead(SCRAPE, title=KRAS)]
    )

    assert scrape_first[0].canonical_url == RECORD
    assert record_first[0].canonical_url == RECORD


def test_a_copy_with_no_identifier_joins_the_registered_record_it_belongs_to():
    merged = merge_leads(
        [],
        [
            _lead(
                "https://pubs.rsc.org/en/content/articlelanding/2014/ta/c4ta00001a",
                title=PAPER,
                identifiers={"doi": "10.1039/c4ta00001a"},
            ),
            _lead("https://www.researchgate.net/publication/299544966", title=PAPER),
        ],
    )

    assert len(merged) == 1
    assert merged[0].identifiers["doi"] == "10.1039/c4ta00001a"


def test_two_registered_records_stay_two_however_their_titles_read():
    """A preprint and its published version are two documents, and so is an erratum
    that reprints the paper's name -- which the corrections facet exists to find."""
    merged = merge_leads(
        [],
        [
            _lead("https://a.org/x", title=PAPER, identifiers={"doi": "10.1039/a"}),
            _lead("https://b.org/y", title=PAPER, identifiers={"doi": "10.1039/b"}),
        ],
    )

    assert len(merged) == 2


def test_a_title_too_short_to_name_a_paper_merges_nothing():
    """Page furniture is what a short title usually is, and two sites share it."""
    merged = merge_leads(
        [],
        [
            _lead("https://a.org/x", title="Cell Reports Medicine"),
            _lead("https://b.org/y", title="Cell Reports Medicine"),
        ],
    )

    assert len(merged) == 2


def test_two_records_in_one_registry_survive_a_title_the_search_wrote():
    """Where the search returns no title, the ingest reads the prose around the link.

    That sentence is not the document's name and several passes can share it, so a
    title match between two records of one registry proves nothing. Seven PubMed
    records of a seven-pass fan-out arrived under one such line.
    """
    merged = merge_leads(
        [],
        [
            _lead(
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title="Supporting contradictory negative null replication methods",
                identifiers={"pmid": pmid},
            )
            for pmid in ("1", "2", "3")
        ],
    )

    assert len(merged) == 3


def test_a_registry_number_is_read_off_the_locator_that_states_it():
    assert stated_identifiers("https://pubmed.ncbi.nlm.nih.gov/33971321/") == {
        "pmid": "33971321"
    }
    assert stated_identifiers("https://ncbi.nlm.nih.gov/pmc/articles/PMC8137619/") == {
        "pmcid": "PMC8137619"
    }
    # v1 and v2 are one paper: arXiv assigns the identifier to the work and the
    # suffix to the upload, so an abstract page and a versioned PDF are one lead.
    assert stated_identifiers("https://arxiv.org/abs/2401.12345") == stated_identifiers(
        "https://arxiv.org/pdf/2401.12345v2"
    )


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


def test_a_title_the_publisher_set_in_html_is_printed_as_text_and_not_as_tags():
    """ACS sets its titles in HTML and the search stored the markup whole.

    Two entries of a live reference list drew their tags on the page: "16. Degradation
    Effects in Li<sub>4</sub>Ti<sub>5</sub>O<sub>12</sub>-Based Cells-Learning from
    Electrode Potential Profiles (2024)" and "18. One-Step Integrated Comodification to
    Improve the Electrochemical Performances of High-Voltage LiCoO<sub>2</sub> for
    Lithium-Ion Batteries (2020)". Flat digits, because Al2O3 and NMC811 are how the
    rest of the report writes a formula.
    """
    titanate = _lead(
        "https://pubs.acs.org/doi/10.1021/acsami.4c12988",
        title=(
            "Degradation Effects in Li<sub>4</sub>Ti<sub>5</sub>O<sub>12</sub>-Based "
            "Cells-Learning from Electrode Potential Profiles"
        ),
    )
    cobaltate = _lead(
        "https://pubs.acs.org/doi/10.1021/acssuschemeng.0c02308",
        title=(
            "One-Step Integrated Comodification to Improve the Electrochemical "
            "Performances of High-Voltage LiCoO<sub>2</sub> for Lithium-Ion Batteries"
        ),
    )
    # Short enough that the chrome cut hands its fallback back: the tags still go.
    terse = _lead("https://pubs.acs.org/doi/10.1021/x", title="Li<sub>2</sub>O anodes")

    assert _reference_title(titanate) == (
        "Degradation Effects in Li4Ti5O12-Based Cells-Learning from Electrode "
        "Potential Profiles"
    )
    assert _reference_title(cobaltate) == (
        "One-Step Integrated Comodification to Improve the Electrochemical "
        "Performances of High-Voltage LiCoO2 for Lithium-Ion Batteries"
    )
    assert _reference_title(terse) == "Li2O anodes"


def test_a_less_than_sign_in_a_title_is_not_mistaken_for_a_tag():
    """A bracket followed by a space and a number opens nothing."""
    lead = _lead(
        "https://example.org/a",
        title="Dendrite suppression at T < 100 K in solid electrolytes",
    )

    assert _reference_title(lead) == (
        "Dendrite suppression at T < 100 K in solid electrolytes"
    )


def test_a_locator_that_spells_the_paper_out_names_the_entry_the_search_did_not():
    """Eight of the twenty-four entries of a live reference list read "Untitled source
    on <host>". Six of those eight locators carried the document's name in their own
    path, so the list withheld from the reader a name it was already holding."""
    researchgate = _lead(
        "https://www.researchgate.net/publication/299544966_Ultrathin_Al2O3_Coatings_"
        "for_Improved_Cycling_Performance_and_Thermal_Stability_of_"
        "LiNi05Co02Mn03O2_Cathode_Material"
    )
    aip = _lead(
        "https://pubs.aip.org/aip/apr/article/8/3/031301/124962/"
        "Understanding-the-roles-of-atomic-layer-deposition",
        title="www.pubs.aip.org",
    )
    blog = _lead(
        "https://blog.epectec.com/why-dot-un-38.3-is-required-for-lithium-batteries"
    )

    # The catalogue's record number in front of the name is the catalogue's, and the
    # slug's own capitals are the title's and are kept.
    # And the points the path dropped out of the composition are put back: the entry
    # shipped "LiNi05Co02Mn03O2", which is not a compound anyone could look up.
    assert _reference_title(researchgate) == (
        "Ultrathin Al2O3 Coatings for Improved Cycling Performance and Thermal "
        "Stability of LiNi0.5Co0.2Mn0.3O2 Cathode Material"
    )
    assert _reference_title(aip) == "Understanding the roles of atomic layer deposition"
    assert _reference_title(blog) == (
        "Why dot un 38.3 is required for lithium batteries"
    )


def test_only_a_formula_gets_its_decimal_points_back_from_the_address():
    """A path cannot carry a point, and one kind of word is missing one for certain.

    A zero between an element symbol and another digit opened a fractional subscript,
    since no compound writes an integer one with a leading zero. Nothing else in a
    title is treated that way: a word needs three element groups to be read as a
    formula at all, so an initialism with a number after it is left as it was.
    """
    formula = _lead(
        "https://example.org/papers/"
        "Thermal-runaway-of-LiNi08Mn01Co01O2-under-abuse-conditions"
    )
    report = _lead(
        "https://example.org/papers/"
        "Why-the-CO2-and-NMC811-figures-in-report-PR2024-are-not-comparable"
    )

    assert _reference_title(formula) == (
        "Thermal runaway of LiNi0.8Mn0.1Co0.1O2 under abuse conditions"
    )
    assert _reference_title(report) == (
        "Why the CO2 and NMC811 figures in report PR2024 are not comparable"
    )


def test_a_locator_carrying_an_identifier_rather_than_a_name_still_says_untitled():
    """The other two of the eight, plus the two the rule has to refuse: a path is only
    read as a name where it reads as prose, which one function word in it settles.
    "PR2024_701_Ihala-Gamaralalage_Chanaka_Safety-Reliability" is a report number and
    three surnames, and "lithium-ion-batteries-ald-coatings-forge-nano" is a topic
    followed by the site's own name -- neither is what the paper is called."""
    numeric = _lead("https://www.mdpi.com/2313-0105/11/6/209")
    authors = _lead(
        "https://www.sandia.gov/app/uploads/sites/82/2024/08/"
        "PR2024_701_Ihala-Gamaralalage_Chanaka_Safety-Reliability.pdf"
    )
    site_name = _lead(
        "http://www.forgenano.com/archivesite/"
        "lithium-ion-batteries-ald-coatings-forge-nano/"
    )

    assert _reference_title(numeric) == "Untitled source on mdpi.com"
    assert _reference_title(authors) == "Untitled source on sandia.gov"
    assert _reference_title(site_name) == "Untitled source on forgenano.com"


def test_a_journal_where_the_papers_name_should_be_gives_way_to_the_address():
    """Entry 5 of that list read "Nano Energy - Western Engineering (2017)" -- a
    journal and the site it was found on -- over a locator whose path spelled the
    paper out. Neither chrome cut catches it: "Western Engineering" is not the
    hostname, eng.uwo.ca does not prove it the site's name either, and what a cut
    would leave is two words. Asserting a journal's name as a paper's is worse than
    saying nothing."""
    lead = _lead(
        "https://www.eng.uwo.ca/nanoenergy/publications/2017/PDFs/New-Insight-into-"
        "Stable-Protective-Layer-for-Long-life-and-Safe-High-Voltage-Cathodes.pdf",
        title="Nano Energy - Western Engineering",
        year="2017",
    )

    assert _reference_title(lead) == (
        "New Insight into Stable Protective Layer for Long life and Safe High Voltage "
        "Cathodes (2017)"
    )


def test_a_path_that_writes_its_spaces_out_as_percent_twenty_is_still_a_name():
    """Entry 27 of a live reference list read "Materials Chemistry A" -- the journal,
    asserted as the paper's name -- over a path that spelled the paper out in full.
    A publisher that files papers under their names has to encode the spaces, and
    undecoded the whole title is one word: too few to be read as a name, so the
    address never got its turn and the journal stood. The comma the filename ran
    into the author's surname is the space the filename could not spare."""
    encoded = _lead(
        "https://arradiance.com/papers/2020%20UBC,%20Stable%20Zn%20Metal%20Anode%20"
        "Enabled%20by%20ALD%20Al2O3%20for%20Aqueous%20Zinc-ion%20Batteries,Liu.pdf",
        title="Materials Chemistry A",
    )

    assert _reference_title(encoded) == (
        "UBC, Stable Zn Metal Anode Enabled by ALD Al2O3 for Aqueous Zinc ion "
        "Batteries, Liu"
    )


def test_a_short_title_the_address_agrees_with_is_a_short_title_and_is_kept():
    """The rule turns on the two being different names, not on the title being short.
    A locator that repeats the name it was given has confirmed it, not displaced it,
    and a title of four words the address knows nothing about is the only case."""
    agreeing = _lead(
        "https://www.nature.com/articles/silicon-anodes-in-the-last-decade",
        title="Silicon anodes",
    )
    long_enough = _lead(
        "https://example.org/papers/a-survey-of-machine-learning-in-energy-storage",
        title="Deep learning for battery health",
    )

    assert _reference_title(agreeing) == "Silicon anodes"
    assert _reference_title(long_enough) == "Deep learning for battery health"


def test_an_entry_named_from_its_address_says_that_is_where_the_name_came_from():
    """A slug cannot mark which of its hyphens joined a compound -- "solid-state"
    comes back as two words -- and the path may have been shortened by the server. The
    name is worth printing and is not what the document calls itself, so the entry
    that prints it says so, and only that entry does."""
    from coscientist.narrative import _reference_naming

    lead = _lead(
        "https://ecoenergyvista.com/electric-vehicles/"
        "when-were-solid-state-batteries-invented-the-surprising-1970"
    )
    title, named_by_address = _reference_naming(lead)

    assert title == "When were solid state batteries invented the surprising 1970"
    assert named_by_address
    line = _reference_line(
        Citation(number=3, title=title, url=lead.canonical_url, named_by_address=True)
    )
    assert "Named from its address, the search having returned no title for it." in line

    titled = _reference_line(Citation(number=4, title=PAPER, url="https://a.org/b"))
    assert "Named from its address" not in titled


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
