"""What a specialist is handed when its search returns redirectors.

Grounding metadata names no document. It returns an opaque redirect, and a live
run showed the cost of passing that straight through: the discovery agent, told
not to cite a link that expires, cited bare domains instead, and the verifier
marked every one of them inaccessible. Following the redirect first is what
turns the same search into citations a reader can open.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from coscientist.evidence import (
    GROUNDING_REDIRECT_MARKER,
    downgrade_unlocatable_sources,
    names_a_document,
    resolve_grounding_urls,
    resolve_manifest_locators,
    resolve_packet_locators,
)
from coscientist.models import (
    DiscoveryManifest,
    EvidencePacket,
    SourceLead,
    SourceRecord,
)

REDIRECT = f"https://vertexaisearch.cloud.google.com/{GROUNDING_REDIRECT_MARKER}/AbC"
OTHER = f"https://vertexaisearch.cloud.google.com/{GROUNDING_REDIRECT_MARKER}/XyZ"
ARTICLE = "https://doi.org/10.1149/2.0011712jes"


def _resolve(handler, urls) -> list[str]:
    async def _run() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await resolve_grounding_urls(urls, client=client)

    return asyncio.run(_run())


def _redirects_to(target: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if GROUNDING_REDIRECT_MARKER in str(request.url):
            return httpx.Response(302, headers={"location": target})
        return httpx.Response(200)

    return handler


def test_a_redirector_is_reported_as_the_document_it_opens():
    assert _resolve(_redirects_to(ARTICLE), [REDIRECT]) == [ARTICLE]


def test_a_direct_link_is_passed_through_untouched():
    """Only redirectors need following; fetching the rest would be a tax."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200)

    assert _resolve(handler, [ARTICLE]) == [ARTICLE]
    assert requested == []


def test_a_redirector_that_cannot_be_followed_is_dropped():
    """A locator that resolves to nothing is worse than no locator at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    assert _resolve(handler, [REDIRECT, ARTICLE]) == [ARTICLE]


def test_a_redirector_that_only_leads_to_another_one_is_dropped():
    assert _resolve(_redirects_to(OTHER), [REDIRECT]) == []


def test_two_redirectors_to_the_same_paper_are_named_once():
    assert _resolve(_redirects_to(ARTICLE), [REDIRECT, OTHER]) == [ARTICLE]


def test_nothing_grounded_means_nothing_fetched():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("resolution should not fetch anything")

    assert _resolve(handler, []) == []


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_every_redirect_status_the_grounding_service_uses_is_followed(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        if GROUNDING_REDIRECT_MARKER in str(request.url):
            return httpx.Response(status, headers={"location": ARTICLE})
        return httpx.Response(200)

    assert _resolve(handler, [REDIRECT]) == [ARTICLE]


# ---------------------------------------------------------------------------
# The packet, which is where the resolution has to land
# ---------------------------------------------------------------------------

ANOTHER = "https://doi.org/10.1149/2.0022712jes"


def _packet(*urls: str) -> EvidencePacket:
    return EvidencePacket(
        question="Can a coating extend cycle life?",
        sources=[
            SourceRecord(id=f"src_{index}", url=url, title=f"Study {index}")
            for index, url in enumerate(urls, start=1)
        ],
    )


def _resolved(handler, packet: EvidencePacket) -> EvidencePacket:
    async def _run() -> EvidencePacket:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await resolve_packet_locators(packet, client=client)

    return asyncio.run(_run())


def test_the_packet_carries_the_document_and_not_the_redirector():
    """The redirect was already being followed, but its answer was appended to
    the specialist's reply as a trailing note after the packet had been written.
    A live run put twenty-seven redirectors into a corpus of forty-one, and the
    locator guard correctly refused every one: forty-one sources found, fourteen
    a reader could open."""
    resolved = _resolved(_redirects_to(ARTICLE), _packet(REDIRECT))

    assert [source.url for source in resolved.sources] == [ARTICLE]
    assert names_a_document(resolved.sources[0].url)


def test_each_source_keeps_its_own_answer():
    """The bulk helper dedupes and drops what it cannot follow, so a shortened
    list cannot be lined up with its input by position."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("AbC"):
            return httpx.Response(302, headers={"location": ARTICLE})
        if str(request.url).endswith("XyZ"):
            return httpx.Response(302, headers={"location": ANOTHER})
        return httpx.Response(200)

    resolved = _resolved(handler, _packet(REDIRECT, OTHER))

    assert [source.url for source in resolved.sources] == [ARTICLE, ANOTHER]


def test_a_redirector_that_will_not_follow_is_left_as_it_is():
    """It is honestly what the search returned. Replacing it with nothing would
    delete the record; the verification guard is what refuses to call it
    verified."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("AbC"):
            raise httpx.ConnectError("no route", request=request)
        if GROUNDING_REDIRECT_MARKER in str(request.url):
            return httpx.Response(302, headers={"location": ANOTHER})
        return httpx.Response(200)

    resolved = _resolved(handler, _packet(REDIRECT, OTHER))

    assert [source.url for source in resolved.sources] == [REDIRECT, ANOTHER]
    assert downgrade_unlocatable_sources(resolved).sources[0].verification_status == (
        "discovered_unverified"
    )


def test_a_packet_with_no_redirector_is_returned_untouched():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("resolution should not fetch anything")

    packet = _packet(ARTICLE)

    assert _resolved(handler, packet) is packet


def _resolved_manifest(handler, manifest: DiscoveryManifest) -> DiscoveryManifest:
    async def _run() -> DiscoveryManifest:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await resolve_manifest_locators(manifest, client=client)

    return asyncio.run(_run())


def _manifest(*urls: str) -> DiscoveryManifest:
    return DiscoveryManifest(
        question="Can a coating extend cycle life?",
        source_leads=[
            SourceLead(canonical_url=url, title=f"Study {index}")
            for index, url in enumerate(urls, start=1)
        ],
    )


def test_deep_research_leads_carry_the_document_and_not_the_redirector():
    """Deep Research reports its sources through the same redirector search does.

    The packet fix was wired only into the search path, so a paid three-pass run
    discovered ninety leads of which not one named a document; the corpus written
    from them carried nine redirectors and the guard downgraded every one.
    """
    resolved = _resolved_manifest(_redirects_to(ARTICLE), _manifest(REDIRECT))

    assert [lead.canonical_url for lead in resolved.source_leads] == [ARTICLE]
    assert names_a_document(resolved.source_leads[0].canonical_url)


def test_a_lead_whose_redirector_will_not_follow_is_left_as_it_is():
    def refuses(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    manifest = _manifest(REDIRECT)
    assert _resolved_manifest(refuses, manifest).source_leads[0].canonical_url == (
        REDIRECT
    )


def test_a_manifest_with_nothing_to_follow_is_returned_unchanged():
    manifest = _manifest(ARTICLE)

    def unused(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("nothing should have been fetched")

    assert _resolved_manifest(unused, manifest) is manifest


def test_one_paper_cited_twelve_times_is_one_lead_once_the_links_are_followed():
    """Deep Research mints a fresh redirector token per citation.

    So a paper cited twelve times arrives as twelve distinct URLs and survives
    every dedupe, because until the links are followed nothing knows they are
    the same document. A live run's panel showed one ACS paper twelve times and
    reported fifty-five usable sources for a corpus the evidence floor had
    counted as sixteen.
    """
    manifest = _manifest(REDIRECT, OTHER)
    manifest.source_leads[0].facets = ["supporting"]
    manifest.source_leads[0].originating_passes = [1]
    manifest.source_leads[1].facets = ["contradictory"]
    manifest.source_leads[1].originating_passes = [3]

    resolved = _resolved_manifest(_redirects_to(ARTICLE), manifest)

    assert [lead.canonical_url for lead in resolved.source_leads] == [ARTICLE]
    lead = resolved.source_leads[0]
    # Merged, not dropped: the second citation is why the contradictory facet
    # has a source at all.
    assert lead.facets == ["supporting", "contradictory"]
    assert lead.originating_passes == [1, 3]
    assert resolved.verification_handoff_source_ids == [lead.id]
