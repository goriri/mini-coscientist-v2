"""Retrieval behind source verification.

The stage this module serves reported fourteen sources "inaccessible" in a live
run without one fetch having taken place, because the tool it was given raised
``ImportError`` on every call. So these tests are about what actually reaches the
network and what is honestly claimed afterwards: a DOI is a redirect and has to
be followed, a paywall interstitial is a successful response carrying no article,
a retraction outranks a readable document, and a redirect chosen by a remote
server must never be allowed to walk into this network.

Nothing here opens a socket. Documents and registries are served by an httpx
mock transport, and name resolution is stubbed, so a test failure means the code
changed rather than that a publisher was slow.
"""

from __future__ import annotations

import json

import httpx
import pytest

from coscientist.retrieval import (
    TOOL_TEXT_BUDGET_CHARS,
    FetchedDocument,
    RetrievalOutcome,
    SourceMetadata,
    SourceRetriever,
    _is_public_host,
    assess_source,
    assess_sources,
    extract_doi,
    fetch_source_document,
    html_to_text,
    pdf_title,
    pdf_to_text,
    searchable_title,
    titles_agree,
)

ARTICLE_TEXT = "A 2 nm alumina interphase halved first-cycle capacity loss. " * 24

PUBLIC_ADDRESS = "93.184.216.34"
PRIVATE_HOSTS = {
    "internal.corp": "10.4.1.9",
    "metadata.example": "169.254.169.254",
}


@pytest.fixture(autouse=True)
def _resolvable_test_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every host in these tests without asking a real resolver."""

    def _getaddrinfo(host, *_args, **_kwargs):
        address = PRIVATE_HOSTS.get(host, PUBLIC_ADDRESS)
        return [(2, 1, 6, "", (address, 0))]

    monkeypatch.setattr("coscientist.retrieval.socket.getaddrinfo", _getaddrinfo)


def _html(body: str, *, title: str = "Alumina interphases", meta: str = "") -> str:
    return (
        f"<html><head><title>{title}</title>{meta}</head>"
        f"<body><nav>Site menu</nav><script>tracker()</script>"
        f"<p>{body}</p></body></html>"
    )


def _crossref(title: str, **extra) -> dict:
    message = {
        "title": [title],
        "author": [{"given": "Wei", "family": "Chen"}],
        "issued": {"date-parts": [[2023, 4, 1]]},
        "container-title": ["Nature Energy"],
        "publisher": "Springer Nature",
        "type": "journal-article",
    }
    message.update(extra)
    return {"message": message}


def _transport(routes: dict[str, httpx.Response], registry: dict[str, dict]):
    """A mock transport that serves documents by URL and registries by host."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in registry:
            payload = registry[host]
            if payload is None:
                return httpx.Response(404)
            return httpx.Response(
                200,
                content=json.dumps(payload),
                headers={"content-type": "application/json"},
            )
        if host.endswith(("crossref.org", "openalex.org", "datacite.org", "nih.gov")):
            return httpx.Response(404)
        response = routes.get(str(request.url))
        if response is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers=response.headers,
        )

    return httpx.MockTransport(handler)


def _client(routes, registry=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_transport(routes, registry or {}), follow_redirects=False
    )


def _page(markup: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, text=markup, headers={"content-type": "text/html; charset=utf-8"}
    )


def _redirect(to: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": to})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_a_doi_is_read_from_a_bare_id_a_url_or_a_publisher_path():
    assert extract_doi("10.1000/abc") == "10.1000/abc"
    assert extract_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert extract_doi("", "https://tandfonline.com/doi/full/10.1080/xy-7") == (
        "10.1080/xy-7"
    )
    # A sentence ending in a DOI must not take the full stop with it.
    assert extract_doi("See 10.1000/abc.") == "10.1000/abc"
    assert extract_doi("no identifier here") == ""


def test_an_expanded_acronym_is_the_same_paper_and_a_different_paper_is_not():
    assert titles_agree(
        "Effects of ALD coatings on Li-ion cycle life",
        "Effects of atomic layer deposition coatings on lithium-ion cycle life",
    )
    assert not titles_agree(
        "Effects of ALD coatings on Li-ion cycle life",
        "Thermal runaway propagation in prismatic cells",
    )
    # Discovery recorded no title, so the registry record cannot contradict it.
    assert titles_agree("", "Whatever the registry holds")


def test_page_furniture_is_not_offered_to_a_verifier_as_the_article():
    text, title, meta = html_to_text(
        _html(
            "The coating halved capacity loss.",
            meta='<meta name="citation_doi" content="10.1000/abc">',
        )
    )

    assert "The coating halved capacity loss." in text
    assert "Site menu" not in text
    assert "tracker()" not in text
    assert title == "Alumina interphases"
    assert meta["citation_doi"] == "10.1000/abc"


def test_half_a_parsed_page_beats_none_when_the_markup_is_broken():
    text, _title, _meta = html_to_text("<p>Readable prose<<<>")

    assert "Readable prose" in text


def test_a_paywall_interstitial_is_not_a_readable_document():
    assert not FetchedDocument(url="u", status=200, text="Sign in to continue").readable
    assert FetchedDocument(url="u", status=200, text="x" * 600).readable
    assert not FetchedDocument(url="u", status=403, text="x" * 6000).readable


def test_only_a_name_that_resolves_entirely_outside_this_network_is_public():
    assert _is_public_host("publisher.example")
    assert not _is_public_host("")
    assert not _is_public_host("localhost")
    assert not _is_public_host("metadata.google.internal")
    # The guard is on the resolved address, not the spelling: a public-looking
    # name pointed at the metadata server is the attack it exists for.
    assert not _is_public_host("metadata.example")
    assert not _is_public_host("internal.corp")


def test_a_name_that_does_not_resolve_is_refused_rather_than_attempted(
    monkeypatch: pytest.MonkeyPatch,
):
    def _fails(*_args, **_kwargs):
        raise OSError("no such host")

    monkeypatch.setattr("coscientist.retrieval.socket.getaddrinfo", _fails)
    assert not _is_public_host("publisher.example")


# ---------------------------------------------------------------------------
# What the fetcher will and will not reach
# ---------------------------------------------------------------------------


async def test_a_doi_redirect_is_followed_to_the_document_it_names():
    """The whole reason the previous tool verified nothing: a DOI is a 302."""
    routes = {
        "https://doi.org/10.1000/abc": _redirect(
            "https://publisher.example/articles/abc"
        ),
        "https://publisher.example/articles/abc": _page(_html(ARTICLE_TEXT)),
    }
    async with _client(routes) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://doi.org/10.1000/abc",
            claimed_title="Alumina interphases",
        )

    assert outcome.tier == "verified"
    assert outcome.document.final_url == "https://publisher.example/articles/abc"
    assert "halved first-cycle capacity loss" in outcome.text


async def test_a_redirect_into_this_network_is_refused_at_the_hop_that_names_it():
    """Each hop is chosen by the previous server, so each is checked again."""
    routes = {
        "https://publisher.example/a": _redirect("http://metadata.example/token"),
        "http://metadata.example/token": _page(_html(ARTICLE_TEXT)),
    }
    async with _client(routes) as client:
        outcome = await assess_source(
            client, SourceRetriever(), "https://publisher.example/a"
        )

    assert outcome.tier == "inaccessible"
    assert "Refused to fetch a non-public locator" in outcome.document.error
    assert outcome.text == ""


async def test_a_redirect_loop_ends_in_a_reported_failure_not_a_hang():
    routes = {
        "https://publisher.example/loop": _redirect("https://publisher.example/loop"),
    }
    async with _client(routes) as client:
        outcome = await assess_source(
            client, SourceRetriever(), "https://publisher.example/loop"
        )

    assert outcome.tier == "inaccessible"
    assert "redirects without a document" in outcome.document.error


def _registry_handler(
    *, converter: dict | None = None, search: dict | None = None, works: dict | None
) -> object:
    """A publisher that refuses everything, and the registries behind it."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "idconv" in url:
            return httpx.Response(200, json=converter or {"records": []})
        if "query.bibliographic" in url:
            return httpx.Response(200, json=search or {"message": {"items": []}})
        if request.url.host == "api.crossref.org":
            return httpx.Response(200, json=works or {"message": {}})
        if request.url.host.endswith(("openalex.org", "datacite.org", "nih.gov")):
            return httpx.Response(404)
        return httpx.Response(403, text="Access denied")

    return httpx.MockTransport(handler)


async def test_a_pmc_article_number_reaches_the_registry_its_pubmed_sibling_did():
    """Eleven open-access papers in one run were filed as unreachable.

    The ladder recognised ``pubmed.ncbi.nlm.nih.gov``, whose path is a PMID, and
    not ``pmc.ncbi.nlm.nih.gov``, whose path is an article number. Both name a
    paper every registry will answer for; only one of them was ever asked.
    """
    transport = _registry_handler(
        converter={"records": [{"pmcid": "PMC6641259", "doi": "10.1000/pmc"}]},
        works=_crossref("Boosting the electrochemical performance of a cathode"),
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6641259/",
            claimed_title="Boosting the electrochemical performance of a cathode",
        )

    assert outcome.tier == "metadata_verified"
    assert outcome.metadata.doi == "10.1000/pmc"


async def test_a_title_finds_the_record_a_refusing_publisher_would_not_give_up():
    """A publisher that blocks this fetcher still publishes through Crossref.

    Of the forty-six sources one production run quarantined, none carried a DOI
    in its locator and none sat on a host the ladder had a branch for, so the
    registry tier was never entered once. All forty-six had a title.
    """
    title = "Ultrathin alumina coatings and lithium-ion cycle life"
    transport = _registry_handler(
        search={"message": {"items": [{"DOI": "10.1000/found", "title": [title]}]}},
        works=_crossref(title),
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://www.researchgate.net/publication/370550704_Ultrathin",
            claimed_title=f"{title} | ResearchGate researchgate.net",
        )

    assert outcome.tier == "metadata_verified"
    assert outcome.metadata.doi == "10.1000/found"


async def test_a_search_that_answers_with_a_different_paper_is_refused():
    """A bibliographic search always returns something, and it is a real paper.

    Accepting it unchecked would attach another author's DOI, year and journal
    to this claim -- a citation that resolves, to the wrong work, which is worse
    than the gap it fills.
    """
    transport = _registry_handler(
        search={
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/other",
                        "title": ["Sodium metal anodes in carbonate electrolytes"],
                    }
                ]
            }
        },
        works=_crossref("Sodium metal anodes in carbonate electrolytes"),
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://www.researchgate.net/publication/1_Ultrathin",
            claimed_title="Ultrathin alumina coatings and lithium-ion cycle life",
        )

    assert outcome.tier == "inaccessible"
    assert not outcome.metadata.doi


async def test_a_title_too_thin_to_name_a_paper_is_never_searched_on():
    searches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "query.bibliographic" in str(request.url):
            searches.append(str(request.url))
        if request.url.host.endswith(("crossref.org", "openalex.org", "nih.gov")):
            return httpx.Response(404)
        return httpx.Response(403, text="Access denied")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://publisher.example/a",
            claimed_title="Battery coatings",
        )

    assert searches == []
    assert outcome.tier == "inaccessible"


def test_a_lead_keeps_its_title_and_loses_the_site_it_was_found_on():
    assert (
        searchable_title("Ultrathin AlPO4 coatings | PNAS pnas.org")
        == "Ultrathin AlPO4 coatings"
    )
    # The hyphens inside the words are not the separator, and a pattern that
    # took them for one cut this title off after "diva".
    assert (
        searchable_title("A Self-Driving Lab for Electrolytes - DiVA diva-portal.org")
        == "A Self-Driving Lab for Electrolytes"
    )
    # Nothing that looks like a host, so nothing to take off.
    assert (
        searchable_title("Boosting Li1.2Mn0.54O2 by ALD")
        == "Boosting Li1.2Mn0.54O2 by ALD"
    )


async def test_two_claims_on_one_paper_cost_one_fetch():
    fetches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host.endswith(("crossref.org", "openalex.org", "nih.gov")):
            return httpx.Response(404)
        fetches.append(str(request.url))
        return _page(_html(ARTICLE_TEXT))

    retriever = SourceRetriever()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        for _ in range(2):
            await assess_source(client, retriever, "https://publisher.example/a")

    assert fetches == ["https://publisher.example/a"]


async def test_the_document_budget_stops_a_manifest_becoming_a_crawl():
    routes = {
        f"https://publisher.example/{index}": _page(_html(ARTICLE_TEXT))
        for index in range(4)
    }
    retriever = SourceRetriever(max_documents=2)
    async with _client(routes) as client:
        outcomes = [
            await assess_source(client, retriever, url) for url in sorted(routes)
        ]

    assert [outcome.tier for outcome in outcomes] == [
        "verified",
        "verified",
        "inaccessible",
        "inaccessible",
    ]
    assert "document budget" in outcomes[-1].document.error


async def test_a_registry_lookup_may_not_be_pointed_at_an_arbitrary_host():
    """The registry path takes URLs built from source data, so it is fenced."""
    async with _client({}) as client:
        with pytest.raises(ValueError, match="approved hosts"):
            await SourceRetriever()._registry_json(
                client, "https://publisher.example/works/10.1000/abc"
            )


# ---------------------------------------------------------------------------
# What may honestly be claimed afterwards
# ---------------------------------------------------------------------------


async def test_a_paywalled_paper_a_registry_confirms_is_confirmed_not_lost():
    """The tier that stops the evidence floor measuring open-access availability."""
    routes = {
        "https://doi.org/10.1000/abc": _redirect("https://publisher.example/abc"),
        "https://publisher.example/abc": _page("<p>Sign in to read this article.</p>"),
    }
    registry = {"api.crossref.org": _crossref("Alumina interphases on silicon anodes")}
    async with _client(routes, registry) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://doi.org/10.1000/abc",
            claimed_title="Alumina interphases on silicon anodes",
        )

    assert outcome.tier == "metadata_verified"
    assert outcome.metadata.authors == ["Wei Chen"]
    assert outcome.metadata.year == 2023
    assert outcome.metadata.container == "Nature Energy"
    assert "crossref confirm this record" in outcome.reason


async def test_a_doi_that_resolves_to_a_different_paper_is_quarantined():
    routes = {"https://doi.org/10.1000/abc": _page("<p>Sign in.</p>")}
    registry = {"api.crossref.org": _crossref("Thermal runaway in prismatic cells")}
    async with _client(routes, registry) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://doi.org/10.1000/abc",
            claimed_title="Alumina interphases on silicon anodes",
        )

    assert outcome.tier == "inaccessible"
    assert "is a different document" in outcome.reason


async def test_a_retraction_outranks_a_document_that_reads_perfectly_well():
    routes = {"https://doi.org/10.1000/abc": _page(_html(ARTICLE_TEXT))}
    registry = {
        "api.crossref.org": _crossref("Alumina interphases"),
        "api.openalex.org": {"id": "W1", "is_retracted": True},
    }
    async with _client(routes, registry) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://doi.org/10.1000/abc",
            claimed_title="Alumina interphases",
        )

    assert outcome.tier == "retracted"
    assert "retracted" in outcome.reason.lower()


async def test_an_open_access_copy_is_tried_when_the_publisher_refuses():
    routes = {
        "https://doi.org/10.1000/abc": _page("<p>403</p>", status=403),
        "https://repository.example/abc.html": _page(_html(ARTICLE_TEXT)),
    }
    registry = {
        "api.crossref.org": _crossref("Alumina interphases"),
        "api.openalex.org": {
            "id": "W1",
            "best_oa_location": {"pdf_url": "https://repository.example/abc.html"},
        },
    }
    async with _client(routes, registry) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://doi.org/10.1000/abc",
            claimed_title="Alumina interphases",
        )

    assert outcome.tier == "verified"
    assert "halved first-cycle capacity loss" in outcome.text


async def test_a_pubmed_id_is_confirmed_without_a_doi():
    routes = {"https://pubmed.ncbi.nlm.nih.gov/28001/": _page("<p>Abstract only.</p>")}
    registry = {
        "eutils.ncbi.nlm.nih.gov": {
            "result": {
                "28001": {
                    "title": "Alumina interphases on silicon anodes",
                    "authors": [{"name": "Chen W"}],
                    "pubdate": "2023 Apr",
                    "fulljournalname": "Nature Energy",
                    "elocationid": "doi: 10.1000/abc",
                }
            }
        }
    }
    async with _client(routes, registry) as client:
        outcome = await assess_source(
            client,
            SourceRetriever(),
            "https://pubmed.ncbi.nlm.nih.gov/28001/",
            claimed_title="Alumina interphases on silicon anodes",
        )

    assert outcome.tier == "metadata_verified"
    assert outcome.metadata.identifiers == {"pmid": "28001", "doi": "10.1000/abc"}
    assert outcome.metadata.registries == ["pubmed"]


async def test_nothing_reached_is_inaccessible_and_says_what_was_tried():
    routes = {"https://publisher.example/missing": _page("gone", status=404)}
    async with _client(routes) as client:
        outcome = await assess_source(
            client, SourceRetriever(), "https://publisher.example/missing"
        )

    assert outcome.tier == "inaccessible"
    assert outcome.reason == "HTTP 404"


async def test_one_locator_raising_costs_that_locator_and_not_the_sweep():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("boom"):
            raise RuntimeError("transport exploded")
        if request.url.host.endswith(("crossref.org", "openalex.org", "nih.gov")):
            return httpx.Response(404)
        return _page(_html(ARTICLE_TEXT))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        outcomes = await assess_sources(
            [
                ("https://publisher.example/boom", ""),
                ("https://publisher.example/good", ""),
            ],
            retriever=SourceRetriever(),
            client=client,
        )

    assert outcomes["https://publisher.example/boom"].tier == "inaccessible"
    assert "Retrieval failed" in outcomes["https://publisher.example/boom"].reason
    assert outcomes["https://publisher.example/good"].tier == "verified"


async def test_the_sweep_makes_no_request_when_there_is_nothing_to_check():
    assert await assess_sources([]) == {}


# ---------------------------------------------------------------------------
# The tool the verifier is given
# ---------------------------------------------------------------------------


async def test_the_tool_reports_the_tier_and_truncates_a_sixty_page_pdf(
    monkeypatch: pytest.MonkeyPatch,
):
    """The full text stays on the outcome for the sweep; the prompt gets a slice."""
    long_text = "x" * (TOOL_TEXT_BUDGET_CHARS + 500)

    async def _assess(_client, _retriever, url, *, claimed_title=""):
        return RetrievalOutcome(
            url=url,
            tier="verified",
            document=FetchedDocument(
                url=url, final_url=url, status=200, text=long_text
            ),
            metadata=SourceMetadata(
                doi="10.1000/abc",
                title="Alumina interphases",
                authors=["Wei Chen"],
                year=2023,
                registries=["crossref"],
            ),
            reason="Retrieved the text.",
        )

    monkeypatch.setattr("coscientist.retrieval.assess_source", _assess)
    result = await fetch_source_document("https://doi.org/10.1000/abc")

    assert result["tier"] == "verified"
    assert result["doi"] == "10.1000/abc"
    assert result["registry_authors"] == ["Wei Chen"]
    assert result["text_truncated"] is True
    assert len(result["text"]) == TOOL_TEXT_BUDGET_CHARS


def _pdf(title: str, body: str = "Full text of the report.") -> bytes:
    """A one-page PDF carrying the given /Title, built without touching the disk."""
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    if title:
        writer.add_metadata({"/Title": title})
    buffer = BytesIO()
    writer.write(buffer)
    assert body  # the fixture's text layer is beside the point of these tests
    return buffer.getvalue()


def test_a_pdf_is_read_for_the_title_it_carries_and_not_only_for_its_text():
    """Entry 22 of a live reference list read "Untitled source on sandia.gov" over a
    PDF this run had retrieved in full, because the parser took the text and left
    the /Title where it found it -- and the entry was then counted among the six
    "retrieved and checked against the document they name"."""
    payload = _pdf("Safety and Reliability of Lithium-Ion Cells Under Abuse")

    assert pdf_title(payload) == (
        "Safety and Reliability of Lithium-Ion Cells Under Abuse"
    )
    # Reading the title costs the text nothing.
    assert pdf_to_text(payload) == ""


def test_an_authoring_tools_placeholder_is_not_taken_for_the_documents_name():
    """A tool fills /Title in when the author does not, and what it fills it in with
    is the file the export came from. Naming the export is not naming the work."""
    assert pdf_title(_pdf("Microsoft Word - PR2024_701_final_v3.docx")) == ""
    assert pdf_title(_pdf("Safety-Reliability-2024-final.pdf")) == ""
    assert pdf_title(_pdf("Untitled")) == ""
    assert pdf_title(_pdf("")) == ""
    # And a fragment too short to be a paper's name leaves the entry untitled.
    assert pdf_title(_pdf("Annual Report")) == ""


def test_a_pdf_that_cannot_be_parsed_costs_neither_its_text_nor_a_traceback():
    assert pdf_title(b"not a pdf at all") == ""
    assert pdf_to_text(b"not a pdf at all") == ""
