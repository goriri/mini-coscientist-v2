"""Iterative Deep Research discovery with deterministic provenance and gates."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .model_catalog import (
    DEFAULT_LANGUAGE,
    language_clause,
    session_language_clause,
    source_language_rule,
)
from .models import (
    CREDITED_STATUSES,
    EVIDENCE_FACETS,
    EVIDENCE_FLOOR_CREDIT,
    EVIDENCE_FLOOR_FACETS,
    FACET_PHRASES,
    MAX_DISCOVERY_PASSES,
    METADATA_VERIFIED_WEIGHT,
    VERIFIED_STATUSES,
    DeepResearchRun,
    DiscoveryCoverage,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EnrichmentRequest,
    EvidenceClaim,
    EvidenceFloor,
    EvidencePacket,
    ResearchGap,
    ResearchPlan,
    Session,
    SourceLead,
    SourceRecord,
)
from .normalization import strip_unstorable_characters
from .retrieval import RetrievalOutcome, SourceRetriever, assess_sources

DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
DEEP_RESEARCH_LOCATION = "global"
MAX_DEEP_RESEARCH_PASSES = MAX_DISCOVERY_PASSES
MIN_COVERAGE_IMPROVEMENT = 0.05
SUFFICIENT_COVERAGE = 0.90
DEFAULT_COST_PER_PASS_USD = 3.0
DEFAULT_COST_WARNING_USD = 12.0
DEFAULT_COST_LIMIT_USD = 24.0
"""The hard spend ceiling for one run's discovery, in US dollars.

Eight passes at three dollars each: the seven-facet fan-out plus one
gap-closing pass, and nothing beyond that. The service takes anonymous requests
and a pass cannot be cancelled once started, so this is a bound the code
enforces rather than a budget somebody watches.
"""

FAN_OUT_FACETS = EVIDENCE_FACETS
"""Which facets get a Deep Research interaction of their own on the first wave.

Sequential gap-directed passes were smarter per dollar and slower by a factor of
seven, and they only ever asked the second question after the first came back.
Fanning out across the facets asks all seven at once; the gap-closing pass that
follows is what keeps the gap-direction that the fan-out gives up.
"""
DEFAULT_PASS_TIMEOUT_SECONDS = 3900
MAX_ENRICHMENT_REQUESTS = 6
MAX_REGISTRY_REQUESTS = 30
MAX_RETAINED_SOURCE_LEADS = 90

GROUNDING_REDIRECT_MARKER = "grounding-api-redirect"
"""What a Vertex search-grounding link looks like before it is followed."""

MAX_DISCOVERY_ANGLES = 10
"""How many sub-searches one decomposed discovery pass may fan out into.

Seven of them are the evidence facets, which are fixed; the rest are the plan's
own success criteria. The bus runs four at a time, so this is between two and
three waves of grounded search -- minutes, against a stage that used to take one
call and return one call's worth of literature.
"""

CORPUS_SOURCE_TARGET = 12
"""Distinct sources a whole discovery pass aims for, stated to the specialists.

Two live runs produced four. Nothing was wrong with them: one search was asked
one question and answered it. A target is what makes the shortfall legible to
the agent doing the searching, and it is deliberately a target rather than a
schema minimum -- a contract that rejects a short packet does not produce a
longer one, it produces an invented one.
"""

ANGLE_SOURCE_TARGET = 3
"""Distinct sources one angle aims for, for the same reason and with the same
caveat: an angle whose literature genuinely does not exist reports that instead.
"""

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
_LIST_MARKER_RE = re.compile(r"^[\s\-*#>]*(?:\d+[.)])?\s*")
_REDIRECTOR_HOST = "vertexaisearch.cloud.google.com"
"""The grounding redirector, which a report prints as a link's title when it has
none. It is never the name of a document."""
# A citation marker, as a Deep Research report writes one: [7], or [2, 5, 9].
_CITATION_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_FACET_KEYWORDS = {
    "supporting": (
        "support",
        "evidence",
        "associated",
        "demonstrate",
        "demonstrated",
    ),
    "contradictory": ("contradict", "conflict", "inconsistent", "disagree"),
    "negative_null": ("negative", "null result", "no effect", "failed"),
    "replication": ("replicat", "reproduc", "validation cohort"),
    "methods": ("method", "assay", "protocol", "measurement", "bias"),
    "safety_governance": ("safety", "toxicity", "ethic", "governance", "risk"),
    "corrections_retractions": (
        "retract",
        "correction",
        "expression of concern",
        "withdrawn",
    ),
}
_AUTHORITATIVE_HOST_MARKERS = (
    ".gov",
    ".edu",
    "nature.com",
    "science.org",
    "cell.com",
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "pubmed.ncbi.nlm.nih.gov",
    "doi.org",
    "crossref.org",
    "datacite.org",
    "arxiv.org",
)
_TRACKING_QUERY_KEYS = {
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DeepResearchTransport(Protocol):
    def start(self, *, prompt: str, pass_number: int, session_id: str) -> dict: ...

    def get(self, interaction_id: str) -> dict: ...


class EvidenceStillRunning(RuntimeError):
    """Signal that a short-lived worker should schedule another poll task."""


def resolve_vertex_project() -> str | None:
    """Resolve the Vertex AI project without requiring an API key.

    This mirrors ``agents.configure_vertex_ai_global_endpoint``: an explicit
    ``GOOGLE_CLOUD_PROJECT`` pins the billing project, and Application Default
    Credentials supply it otherwise.  ``None`` means "Vertex is not reachable
    here", which callers treat as a degraded mode rather than an error, so the
    offline package never requires cloud credentials to import or run.
    """
    if project := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return project
    try:
        import google.auth
        from google.auth.exceptions import GoogleAuthError
    except ImportError:
        return None
    try:
        _, project_id = google.auth.default()
    except (GoogleAuthError, OSError):
        # No ADC, or the metadata server is unreachable; both are "unavailable".
        return None
    return project_id or None


class GeminiDeepResearchTransport:
    """Small Interactions API adapter kept separate for deterministic tests.

    Deep Research is reachable through two backends.  Vertex AI with Application
    Default Credentials is preferred because it needs no long-lived secret and
    matches the credentials the rest of the workflow already uses; a Gemini API
    key stays supported for deployers who only have one.  An explicitly passed
    ``api_key`` always wins so a caller can override the machine's ADC.
    """

    backend: str
    project: str | None = None
    location: str | None = None

    def __init__(
        self,
        api_key: str | None = None,
        *,
        project: str | None = None,
        location: str = DEEP_RESEARCH_LOCATION,
    ):
        resolved_project = None if api_key else (project or resolve_vertex_project())
        resolved_key = api_key or (
            None if resolved_project else os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_project and not resolved_key:
            raise RuntimeError(
                "Gemini Deep Research needs either Vertex AI access or an API key, "
                "and neither is available: no Application Default Credentials with "
                "a project (set GOOGLE_CLOUD_PROJECT, or run "
                "`gcloud auth application-default login`) and no GEMINI_API_KEY."
            )

        from google import genai

        if resolved_project:
            self.backend = "vertex"
            self.project = resolved_project
            self.location = location
            self._client = genai.Client(
                vertexai=True, project=resolved_project, location=location
            )
        else:
            self.backend = "api_key"
            self._client = genai.Client(api_key=resolved_key)

    def start(self, *, prompt: str, pass_number: int, session_id: str) -> dict:
        interaction = self._client.interactions.create(
            agent=DEEP_RESEARCH_AGENT,
            input=prompt,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "none",
                "visualization": "off",
                "collaborative_planning": False,
            },
            background=True,
            store=True,
            labels={
                "coscientist_session": session_id[-32:],
                "evidence_pass": str(pass_number),
            },
        )
        return interaction.model_dump(mode="json", exclude_none=True)

    def get(self, interaction_id: str) -> dict:
        interaction = self._client.interactions.get(interaction_id)
        return interaction.model_dump(mode="json", exclude_none=True)


class GeminiEvidenceNormalizer:
    """Grounded extractor using the existing Vertex global Gemini model, no tools."""

    model_id = "gemini-3.1-pro-preview"

    def __init__(self, project: str | None = None):
        from google import genai

        # Application Default Credentials often carry the project without
        # GOOGLE_CLOUD_PROJECT being exported, and passing project=None here
        # raises inside google-genai instead of degrading gracefully.
        resolved_project = project or resolve_vertex_project()
        if not resolved_project:
            raise RuntimeError(
                "Evidence normalization needs a Vertex AI project: set "
                "GOOGLE_CLOUD_PROJECT or configure Application Default Credentials."
            )
        self._client = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=DEEP_RESEARCH_LOCATION,
        )

    def __call__(self, prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=DiscoveryNarrative,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.HIGH,
                    include_thoughts=False,
                ),
                tools=None,
            ),
        )
        return response.text or ""


class EvidenceArtifactStore:
    """Persist raw interactions to GCS when configured; retain a URI otherwise."""

    def __init__(self, bucket_name: str | None = None):
        self.bucket_name = bucket_name or os.environ.get(
            "EVIDENCE_BUCKET_NAME", os.environ.get("LOGS_BUCKET_NAME", "")
        )

    def put(self, session_id: str, pass_number: int, payload: dict) -> str:
        interaction_id = str(payload.get("id") or f"pass-{pass_number}")
        if not self.bucket_name:
            return f"interaction://{interaction_id}"
        from google.cloud import storage

        object_name = (
            f"evidence/{session_id}/deep-research/pass-{pass_number}-"
            f"{interaction_id}.json"
        )
        blob = storage.Client().bucket(self.bucket_name).blob(object_name)
        try:
            blob.upload_from_string(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                content_type="application/json",
                if_generation_match=0,
            )
        except Exception as exc:
            # Cloud Task retries are idempotent: an existing immutable object is
            # success, while all other storage failures remain visible.
            if getattr(exc, "code", None) != 412:
                raise
        return f"gs://{self.bucket_name}/{object_name}"


class RegistryMetadataEnricher:
    """Bounded metadata lookup against fixed scholarly registries."""

    def __init__(
        self, *, max_requests: int = MAX_REGISTRY_REQUESTS, timeout: float = 8
    ):
        self.max_requests = max_requests
        self.timeout = timeout
        self.requests_used = 0
        self._cache: dict[str, dict[str, Any]] = {}

    def _json(self, url: str) -> dict[str, Any]:
        if self.requests_used >= self.max_requests:
            return {}
        if url in self._cache:
            return self._cache[url]
        host = urlsplit(url).hostname
        allowed = {
            "api.crossref.org",
            "api.openalex.org",
            "api.datacite.org",
            "eutils.ncbi.nlm.nih.gov",
        }
        if host not in allowed:
            raise ValueError("Registry enrichment is restricted to approved hosts.")
        self.requests_used += 1
        request = Request(
            url,
            headers={"User-Agent": "mini-coscientist/2 evidence-metadata"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if response.headers.get_content_type() != "application/json":
                    return {}
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    return {}
                value = json.loads(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            value = {}
        self._cache[url] = value if isinstance(value, dict) else {}
        return self._cache[url]

    def enrich(self, leads: list[SourceLead]) -> list[SourceLead]:
        for lead in leads:
            if self.requests_used >= self.max_requests:
                break
            doi = lead.identifiers.get("doi")
            if doi:
                encoded = quote(doi, safe="")
                crossref = self._json(f"https://api.crossref.org/works/{encoded}")
                message = crossref.get("message", {}) if crossref else {}
                titles = message.get("title") or []
                if titles:
                    lead.title = str(titles[0])
                if issued := message.get("issued", {}).get("date-parts", []):
                    if issued and issued[0]:
                        lead.year = int(issued[0][0])
                lead.authors = [
                    " ".join(
                        value
                        for value in (author.get("given", ""), author.get("family", ""))
                        if value
                    )
                    for author in message.get("author", [])
                ][:20]
                openalex_id = quote(f"https://doi.org/{doi}", safe="")
                openalex = self._json(f"https://api.openalex.org/works/{openalex_id}")
                if openalex.get("id"):
                    lead.identifiers["openalex"] = str(openalex["id"])
                datacite = self._json(f"https://api.datacite.org/dois/{encoded}")
                attributes = datacite.get("data", {}).get("attributes", {})
                if attributes.get("publisher") and lead.source_type == "unknown":
                    lead.source_type = "primary_or_authoritative"
            host = urlsplit(lead.canonical_url).hostname or ""
            if host == "pubmed.ncbi.nlm.nih.gov":
                path_parts = [
                    part
                    for part in urlsplit(lead.canonical_url).path.split("/")
                    if part
                ]
                if path_parts and path_parts[0].isdigit():
                    pmid = path_parts[0]
                    lead.identifiers["pmid"] = pmid
                    pubmed = self._json(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                        f"esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
                    )
                    record = pubmed.get("result", {}).get(pmid, {})
                    if record.get("title"):
                        lead.title = str(record["title"])
                    if record.get("pubdate", "")[:4].isdigit():
                        lead.year = int(record["pubdate"][:4])
        return leads


def canonicalize_url(url: str) -> str:
    """Normalize a public locator without dereferencing it."""
    cleaned = url.strip().rstrip(".,;:")
    parts = urlsplit(cleaned)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only absolute HTTP(S) source URLs are allowed.")
    host = parts.hostname.lower().rstrip(".")
    if host in {"localhost", "metadata.google.internal"}:
        raise ValueError("Local and metadata-service URLs are not evidence sources.")
    if host in {
        "google.com",
        "www.google.com",
        "vertexaisearch.cloud.google.com",
        "url.google.com",
    }:
        query_dict = dict(parse_qsl(parts.query))
        for key in ("q", "url", "target", "dest"):
            if key in query_dict and query_dict[key].startswith("http"):
                return canonicalize_url(query_dict[key])
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError(
            "Private, reserved, and link-local URLs are not evidence sources."
        )
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS
    ]
    path = parts.path or "/"
    return urlunsplit(
        (
            parts.scheme.lower(),
            host + (f":{parts.port}" if parts.port else ""),
            path,
            urlencode(filtered_query),
            "",
        )
    )


async def resolve_grounding_urls(urls, *, client, timeout: float = 10.0) -> list[str]:
    """Follow search-grounding redirects to the documents they actually open.

    Grounding metadata never names a source directly. It hands back an opaque
    redirector, and a live run showed exactly what that costs downstream: told
    not to cite a link that expires and names no document, the discovery agent
    fell back to citing bare domains -- "researchgate.net" -- and the verifier
    then marked every one of them inaccessible, because a domain is not a
    document. Neither agent was wrong; they had nothing better to work with.

    Following the redirect here is what gives them something better. One HEAD
    per link, in parallel, and the publisher URL comes back. A redirect that
    cannot be followed is dropped rather than passed on: offering a locator
    that resolves to nothing is the failure this exists to prevent.
    """
    ordered = [url for url in urls]
    if not ordered:
        return []

    async def _follow(url: str) -> str | None:
        if GROUNDING_REDIRECT_MARKER not in url:
            return url
        try:
            response = await client.head(url, follow_redirects=True, timeout=timeout)
        except Exception:
            return None
        final = str(response.url)
        return None if GROUNDING_REDIRECT_MARKER in final else final

    followed = await asyncio.gather(*(_follow(url) for url in ordered))
    return list(dict.fromkeys(url for url in followed if url))


def names_a_document(url: str) -> bool:
    """Whether a locator reaches a document rather than a site or a redirector.

    A bare domain is the failure this catches. It looks like a citation, it
    survives every schema check a URL field can impose, and it names nothing: a
    reader given ``researchgate.net`` has been told which website to search, not
    which paper to read. Discovery is asked in its contract not to emit one, and
    a live run showed it doing so anyway when the grounding metadata gave it
    nothing better, so the rule is enforced here rather than requested there.
    """
    if not url.startswith(("http://", "https://")):
        return False
    if GROUNDING_REDIRECT_MARKER in url:
        return False
    _, _, remainder = url.partition("://")
    _, _, path = remainder.partition("/")
    return bool(path.strip("/"))


async def resolve_packet_locators(packet: EvidencePacket, *, client) -> EvidencePacket:
    """Replace each redirector in a packet with the document it opens.

    The redirect was already being followed, but a stage too late to matter: the
    resolved list was appended to the specialist's answer as a trailing note,
    after the specialist had finished writing the packet, so what each source
    actually carried was still the redirector. A live run put twenty-seven of
    those in a corpus of forty-one, and the guard below correctly refused every
    one of them -- forty-one sources found, fourteen a reader could open.

    Resolving here fixes the record rather than the report. Each link is followed
    on its own, concurrently, because the bulk helper drops what it cannot follow
    and a shortened list cannot be lined up with its input by position. A
    redirect that will not follow is left exactly as it is: it is honestly what
    the search returned, and the guard below will say so.
    """
    followed = await _followed_redirects(
        (source.url for source in packet.sources), client=client
    )
    if not followed:
        return packet
    updated = packet.model_copy(deep=True)
    for source in updated.sources:
        replacement = followed.get(source.url)
        if replacement:
            source.url = replacement
    return updated


async def resolve_manifest_locators(
    manifest: DiscoveryManifest, *, client
) -> DiscoveryManifest:
    """Replace each redirector among a manifest's leads with the document it opens.

    Deep Research reports its sources through the same grounding redirector search
    does, and the fix above was wired only into the search path. A live three-pass
    run therefore discovered ninety leads of which not one named a document, the
    corpus built from them carried nine redirectors, and the guard below downgraded
    every one -- a paid run whose reference list said "no link to this source was
    recorded" nine times. Resolving the manifest first is what fixes the corpus
    too: the packet is written from these leads.

    The leads are then re-merged, because following the links is what reveals
    which of them are the same source. Deep Research mints a fresh redirector
    token for every citation it prints, so one paper cited twelve times arrives
    as twelve distinct URLs and survives every dedupe until they are followed.
    Un-merged, that paper filled twelve rows of the evidence panel, and the
    panel's "fifty-five usable sources" stood next to an evidence floor that had
    counted the same corpus as sixteen.
    """
    followed = await _followed_redirects(
        (lead.canonical_url for lead in manifest.source_leads), client=client
    )
    if not followed:
        return manifest
    updated = manifest.model_copy(deep=True)
    for lead in updated.source_leads:
        replacement = followed.get(lead.canonical_url)
        if replacement:
            lead.canonical_url = replacement
            # And what the resolved link says the document is. Without this the
            # merge below reads the new address off one copy of a paper and a
            # DOI off the other, calls them two documents, and the panel counts
            # the paper twice.
            lead.identifiers = {
                **stated_identifiers(replacement),
                **lead.identifiers,
            }
    updated.source_leads = merge_leads([], updated.source_leads)
    updated.verification_handoff_source_ids = [lead.id for lead in updated.source_leads]
    return updated


async def _followed_redirects(urls, *, client) -> dict[str, str]:
    """Where each distinct redirector among ``urls`` leads, omitting those that do not.

    Each link is followed on its own, concurrently, because the bulk helper drops
    what it cannot follow and a shortened list cannot be lined up with its input by
    position.
    """
    pending = list(
        dict.fromkeys(url for url in urls if GROUNDING_REDIRECT_MARKER in url)
    )
    if not pending:
        return {}

    async def _one(url: str) -> tuple[str, str | None]:
        opened = await resolve_grounding_urls([url], client=client)
        return url, opened[0] if opened else None

    return {
        url: opened
        for url, opened in await asyncio.gather(*map(_one, pending))
        if opened
    }


def downgrade_unlocatable_sources(packet: EvidencePacket) -> EvidencePacket:
    """Refuse a verified status to any source whose locator names no document.

    Verification means someone opened the document and found the claim at the
    location they recorded. That is not a thing anyone can have done with a
    publisher's front page, so a packet asserting it is asserting something
    impossible -- and nothing downstream can tell that entry from a real one,
    because all the report reads is the status field.

    Downgraded rather than dropped: the search did find something, and deleting
    the record hides the gap instead of reporting it. The status becomes
    ``inaccessible``, which is what it is, and the packet says why in its
    limitations so the loss is visible in the report and not only in the diff.
    """
    unlocatable = {
        source.id
        for source in packet.sources
        if not names_a_document(source.url)
        and source.verification_status in {"verified", "corrected"}
    }
    if not unlocatable:
        return packet
    downgraded = packet.model_copy(deep=True)
    for source in downgraded.sources:
        if source.id in unlocatable:
            source.verification_status = "inaccessible"
    for claim in downgraded.claims:
        # A claim is only as verified as the document it rests on. Leaving one
        # verified under a source that is not would put the strongest wording in
        # the report on the weakest footing in the corpus. Nor is an unverified
        # claim exempt: "nobody checked" and "the locator reaches no document" are
        # different findings, and the report says different things about them.
        if claim.source_id in unlocatable and claim.verification_status != "retracted":
            claim.verification_status = "inaccessible"
    downgraded.limitations.append(
        f"{len(unlocatable)} source"
        + ("" if len(unlocatable) == 1 else "s")
        + " reported as verified named a website rather than a document, so "
        + ("it was" if len(unlocatable) == 1 else "they were")
        + " recorded as inaccessible instead: a locator that reaches no document "
        "cannot have been checked against one."
    )
    return downgraded


_TIER_RANK = {
    "inaccessible": 0,
    "discovered_unverified": 0,
    "metadata_verified": 1,
    "verified": 2,
    "corrected": 2,
}


async def sweep_verification(
    packet: EvidencePacket, *, retriever: SourceRetriever | None = None
) -> EvidencePacket:
    """Cap every status in a packet at what an actual retrieval supports.

    The specialist decides whether a document says what was attributed to it,
    because that is a judgement about meaning. It does not get to decide whether
    the document was read, because that is a fact, and a run in which the only
    retrieval tool raised ``ImportError`` on every call still returned a packet
    full of confident statuses. Nothing downstream could tell those from real
    ones: all the report reads is the status field.

    So each locator is fetched here, independently and concurrently, and the
    result sets a ceiling. A source whose text arrived keeps whatever the
    specialist concluded. One that only a registry could confirm cannot be
    called verified. One that neither reached is inaccessible whatever the
    packet says. A retraction overrides everything, in the other direction:
    a retracted paper is perfectly readable and must never be cited as support.

    Registry metadata is written back at the same time, so the reference list
    can show authors and a year instead of a URL.
    """
    if not packet.sources:
        return packet
    targets = [
        (source.url, source.title)
        for source in packet.sources
        if names_a_document(source.url)
    ]
    outcomes = await assess_sources(targets, retriever=retriever)
    if not outcomes:
        return packet
    return apply_retrieval_outcomes(packet, outcomes)


def _document_title(title: str) -> str:
    """A retrieved document's own title, where it is long enough to be one.

    A publisher's ``<title>`` is as often the site as the paper -- "ScienceDirect",
    "PubMed" -- and asserting a site's name as a paper's is worse than saying
    nothing. Four words is the floor, below which the reference list keeps its
    "Untitled source on <host>" and the locator gets its turn at naming the entry.
    """
    cleaned = " ".join(title.split())
    return cleaned if len(cleaned.split()) >= 4 else ""


def apply_retrieval_outcomes(
    packet: EvidencePacket, outcomes: dict[str, RetrievalOutcome]
) -> EvidencePacket:
    """Reconcile a packet against retrieval results. Pure, so it can be tested."""
    updated = packet.model_copy(deep=True)
    demoted: list[str] = []
    retracted: list[str] = []
    for source in updated.sources:
        outcome = outcomes.get(source.url)
        if outcome is None:
            if not names_a_document(source.url):
                source.verification_status = "inaccessible"
                source.verification_note = (
                    "This locator names a website rather than a document, so no "
                    "retrieval was attempted."
                )
            continue
        metadata = outcome.metadata
        if not source.title:
            # The registry holds the title of record and answers first, but a source
            # with no DOI has no registry entry -- and the document itself carries
            # one, in its <title> element or its PDF /Title. Retrieval read it for
            # its text and never for that, so a paper this run had in full reached
            # the reference list as "Untitled source on sandia.gov" and was counted
            # among those "checked against the document they name".
            source.title = metadata.title or _document_title(outcome.document.title)
        if metadata.authors:
            source.authors = metadata.authors
        if metadata.year:
            source.year = metadata.year
        if metadata.container:
            source.container = metadata.container
        if metadata.identifiers:
            source.identifiers = {**metadata.identifiers, **source.identifiers}
        source.verification_note = outcome.reason
        if outcome.tier == "retracted":
            source.verification_status = "retracted"
            retracted.append(source.id)
            continue
        ceiling = _TIER_RANK[outcome.tier]
        if _TIER_RANK.get(source.verification_status, 0) > ceiling:
            source.verification_status = outcome.tier
            demoted.append(source.id)
        elif (
            source.verification_status == "discovered_unverified"
            and outcome.tier == "metadata_verified"
        ):
            # The specialist never reached a conclusion about this one, and the
            # registry did. Recording what is known beats recording nothing.
            source.verification_status = "metadata_verified"
        elif (
            source.verification_status == "discovered_unverified"
            and outcome.tier == "inaccessible"
        ):
            # Same tie, the other way round: both words rank 0, so the ceiling test
            # above leaves the source saying nobody looked when the run did look and
            # the locator did not open. The reference entry prints one of those two
            # sentences off this field.
            source.verification_status = "inaccessible"
    ceilings = {source.id: source.verification_status for source in updated.sources}
    for claim in updated.claims:
        status = ceilings.get(claim.source_id or "")
        if status is None:
            continue
        if status == "retracted":
            claim.verification_status = "retracted"
        elif status == "inaccessible" and claim.verification_status != "retracted":
            # "inaccessible" and "discovered_unverified" both rank 0, so the ceiling
            # test below never fired between them and a claim resting on a document
            # this run went back to and could not open kept the weaker of the two
            # words. They are not the same finding: one says nobody tried, the other
            # says someone did and failed. A live report badged such a statement
            # [Literature Lead] -- there is something here to follow -- eighty lines
            # above the reference entry for that same document reading "Could not be
            # retrieved when this run went back to it. Nothing here is grounded by
            # it." Propagated like "retracted" for the same reason it is.
            claim.verification_status = "inaccessible"
        elif _TIER_RANK.get(claim.verification_status, 0) > _TIER_RANK.get(status, 0):
            # A claim is only as verified as the document it rests on.
            claim.verification_status = status
    if demoted:
        updated.limitations.append(
            f"{len(demoted)} source"
            + ("" if len(demoted) == 1 else "s")
            + " claimed a stronger verification status than retrieval supported "
            + ("and was" if len(demoted) == 1 else "and were")
            + " downgraded to what the fetch and the scholarly registries could "
            "actually establish."
        )
    if retracted:
        updated.limitations.append(
            f"{len(retracted)} source"
            + ("" if len(retracted) == 1 else "s")
            + " in this corpus "
            + ("is" if len(retracted) == 1 else "are")
            + " recorded as retracted by a scholarly registry and must not be "
            "cited as support."
        )
    return updated


def evaluate_evidence_floor(
    packet: EvidencePacket,
    manifest: DiscoveryManifest | None = None,
) -> EvidenceFloor:
    """Whether this corpus is strong enough to generate hypotheses from.

    The test it replaces demanded that every source in the packet be verified,
    which is unclearable by construction: one publisher being down on the day
    fails a corpus of ninety. This measures the three things that actually
    decide whether a hypothesis rests on a literature -- how much was checked,
    how many kinds of evidence were found, and whether anything was found that
    disagrees -- and reports each of them whether or not the floor is met, since
    the researcher is being asked to decide, not merely informed of a verdict.
    """
    credited = [
        source
        for source in packet.sources
        if source.verification_status in CREDITED_STATUSES
    ]
    verified = [
        source for source in credited if source.verification_status in VERIFIED_STATUSES
    ]
    metadata_only = [
        source
        for source in credited
        if source.verification_status == "metadata_verified"
    ]
    credit = len(verified) + METADATA_VERIFIED_WEIGHT * len(metadata_only)

    facets_by_url = {}
    for lead in manifest.source_leads if manifest else []:
        facets_by_url.setdefault(lead.canonical_url, set()).update(lead.facets)
    covered: set[str] = set()
    for source in credited:
        if source.facet:
            covered.add(source.facet)
        covered.update(facets_by_url.get(source.url, ()))
    covered &= set(EVIDENCE_FACETS)

    contradicting_source_ids = {
        claim.source_id
        for claim in packet.claims
        if claim.relation == "contradicts" and claim.source_id
    }
    disconfirming = sum(source.id in contradicting_source_ids for source in credited)
    searched = bool(
        {"contradictory", "negative_null"}
        & (
            set(manifest.discovery_angles if manifest else ())
            | {run.facet for run in (manifest.runs if manifest else ())}
        )
    )

    floor = EvidenceFloor(
        verified_sources=len(verified),
        metadata_verified_sources=len(metadata_only),
        weighted_credit=round(credit, 2),
        facets_covered=sorted(covered),
        facets_missing=[facet for facet in EVIDENCE_FACETS if facet not in covered],
        disconfirming_sources=disconfirming,
        retracted_sources=sum(
            source.verification_status == "retracted" for source in packet.sources
        ),
        inaccessible_sources=sum(
            source.verification_status in {"inaccessible", "discovered_unverified"}
            for source in packet.sources
        ),
        searched_for_disconfirming=searched,
    )
    if not floor.credit_met:
        floor.shortfalls.append(
            f"{floor.weighted_credit:g} of {EVIDENCE_FLOOR_CREDIT:g} weighted "
            f"verified sources: {floor.verified_sources} read in full and "
            f"{floor.metadata_verified_sources} confirmed by a registry but "
            "unreadable, which count for half each."
        )
    if not floor.facets_met:
        floor.shortfalls.append(
            f"{len(floor.facets_covered)} of {EVIDENCE_FLOOR_FACETS} required "
            "evidence facets have a verified source. Missing: "
            + ", ".join(
                FACET_PHRASES.get(facet, facet.replace("_", " "))
                for facet in floor.facets_missing
            )
            + "."
        )
    if not disconfirming and not searched:
        # Soft only once the search has happened. "We found none" is a finding
        # about the literature; "we never looked" is a hole in the method.
        floor.shortfalls.append(
            "No search was run for contradictory or negative evidence, so the "
            "absence of any is not yet a finding."
        )
    floor.met = bool(
        floor.credit_met and floor.facets_met and (disconfirming or searched)
    )
    return floor


@dataclass(frozen=True)
class DiscoveryAngle:
    """One sub-search of a decomposed discovery pass."""

    key: str
    brief: str


def discovery_angles(plan: ResearchPlan) -> tuple[DiscoveryAngle, ...]:
    """Break one research question into the searches that would answer it.

    A single grounded search returns a single search's worth of literature, and
    two live runs bear that out: the whole knowledge base was four sources and
    six claims, every one of them supporting. That is not the model failing at
    what it was asked. Nobody asked it for the contradicting study.

    So the pass is decomposed. Each angle is a search a competent reviewer would
    run before believing an answer, they are dispatched concurrently, and their
    packets are merged. The facets come first because they are the axes coverage
    is scored on, and the plan's own success criteria follow, because those are
    what this particular question has to be able to show.
    """
    angles = [
        DiscoveryAngle(
            key=facet,
            brief=f"Search specifically for {FACET_PHRASES[facet]}.",
        )
        for facet in EVIDENCE_FACETS
    ]
    angles.extend(
        DiscoveryAngle(
            key=f"criterion_{index}",
            brief=(
                "Search for the literature that would establish or refute this "
                f"success criterion: {criterion}"
            ),
        )
        for index, criterion in enumerate(plan.success_criteria[:3], start=1)
    )
    return tuple(angles[:MAX_DISCOVERY_ANGLES])


def _unclaimed_id(preferred: str, used: set[str]) -> str:
    """Keep the identifier the specialist chose unless another angle took it.

    Renumbering everything would be simpler and worse. ``src_alumina`` says what
    it is; ``src_003`` says where it landed in a merge, and the identifiers are
    what a reader follows from a hypothesis back to the paper behind it. So the
    original stands, and a collision -- two angles that each called something
    ``src_1`` -- is the only thing that moves.
    """
    candidate, suffix = preferred, 2
    while candidate in used:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def merge_evidence_packets(
    question: str, packets: list[EvidencePacket]
) -> EvidencePacket:
    """Fold the per-angle packets into the one corpus the stage hands on.

    Two angles that find the same paper must not put it in the corpus twice, and
    two that find different papers must not collide on an identifier: each
    packet is written independently, so ``src_1`` in one has nothing to do with
    ``src_1`` in the next. Sources are keyed on the canonical URL and the ids are
    rewritten, with every claim's ``source_id`` following its source.
    """
    merged_sources: dict[str, SourceRecord] = {}
    remapped: dict[tuple[int, str], str] = {}
    used_ids: set[str] = set()
    for index, packet in enumerate(packets):
        for source in packet.sources:
            try:
                key = canonicalize_url(source.url)
            except ValueError:
                # An unusable locator is still a record of what the search saw.
                # Keyed on its own text, so it neither collides nor merges.
                key = source.url.strip().lower()
            existing = merged_sources.get(key)
            if existing is None:
                copy = source.model_copy(deep=True)
                copy.id = _unclaimed_id(source.id, used_ids)
                copy.supports_claim_ids = []
                merged_sources[key] = copy
                remapped[(index, source.id)] = copy.id
                continue
            remapped[(index, source.id)] = existing.id
            if not existing.title and source.title:
                existing.title = source.title
            if existing.source_type == "unknown":
                existing.source_type = source.source_type

    merged_claims: list[EvidenceClaim] = []
    seen: set[str] = set()
    for index, packet in enumerate(packets):
        for claim in packet.claims:
            fingerprint = " ".join(claim.claim.lower().split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            copy = claim.model_copy(deep=True)
            copy.id = _unclaimed_id(claim.id, used_ids)
            copy.source_id = remapped.get((index, claim.source_id or ""))
            merged_claims.append(copy)

    by_id = {source.id: source for source in merged_sources.values()}
    for claim in merged_claims:
        source = by_id.get(claim.source_id or "")
        if source is not None:
            source.supports_claim_ids.append(claim.id)

    limitations: list[str] = []
    for packet in packets:
        limitations.extend(
            item for item in packet.limitations if item not in limitations
        )
    return EvidencePacket(
        question=question,
        sources=list(merged_sources.values()),
        claims=merged_claims,
        limitations=limitations,
    )


def _content_text(value: Any) -> str:
    """Flatten an Interactions content field into plain text.

    Vertex returns ``steps[].content`` as a list of typed parts such as
    ``{"type": "text", "text": ..., "annotations": [...]}`` while other shapes
    return a bare string, so both are accepted rather than assumed.

    Text is stripped of the characters PostgreSQL will not store, here at the
    one door every scrap of Deep Research prose comes through, because what is
    downstream of it is a specialist call whose event log is a Postgres row.
    """
    if isinstance(value, str):
        return strip_unstorable_characters(value) if value.strip() else ""
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return strip_unstorable_characters(text)
        return ""
    if isinstance(value, list):
        parts = [_content_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _extract_report(payload: dict) -> str:
    """Return the final Deep Research report from either backend's payload."""
    if text := _content_text(payload.get("output_text")):
        return text
    for step in reversed(payload.get("steps") or []):
        if not isinstance(step, dict):
            continue
        if step.get("type") in {"thought", "user_input"}:
            # Reasoning summaries and the echoed prompt are not the report.
            continue
        for key in ("content", "output", "result"):
            if text := _content_text(step.get(key)):
                return text
    return ""


def _citation_titles(value: Any) -> dict[str, str]:
    """Collect ``url_citation`` annotation titles keyed by their raw URL.

    Vertex Deep Research cites with ``[cite: n]`` markers instead of inline
    Markdown links, so the annotation list is the only place a cited locator
    carries a human-readable title.  Without this, every Vertex-discovered lead
    would be titled with the grounding-redirect hostname.
    """
    titles: dict[str, str] = {}
    if isinstance(value, dict):
        url = value.get("url") or value.get("uri")
        title = value.get("title")
        if (
            value.get("type") == "url_citation"
            and isinstance(url, str)
            and url.startswith(("http://", "https://"))
            and isinstance(title, str)
            and title.strip()
        ):
            titles[url] = " ".join(title.split())[:300]
        for child in value.values():
            titles.update(_citation_titles(child))
    elif isinstance(value, list):
        for child in value:
            titles.update(_citation_titles(child))
    return titles


def _payload_urls(value: Any) -> list[str]:
    if isinstance(value, dict):
        found = [
            child
            for key, child in value.items()
            if key in {"url", "uri"}
            and isinstance(child, str)
            and child.startswith(("http://", "https://"))
        ]
        for child in value.values():
            found.extend(_payload_urls(child))
        return found
    if isinstance(value, list):
        return [url for child in value for url in _payload_urls(child)]
    return []


def _source_leads(
    report: str,
    pass_number: int,
    raw_reference: str,
    *,
    citation_urls: list[str] | None = None,
    citation_titles: dict[str, str] | None = None,
) -> list[SourceLead]:
    titles = dict(citation_titles or {})
    for title_text, raw_link in _MARKDOWN_LINK_RE.findall(report):
        clean_title = title_text.strip()
        # What a report prints when it has no title for a link is the grounding
        # redirector's own hostname, and taking it leaves every lead from that
        # pass called "vertexaisearch.cloud.google.com".
        if not clean_title or clean_title == _REDIRECTOR_HOST:
            continue
        titles[raw_link] = clean_title
        # Keyed both ways: the loop below looks the title up by canonical URL,
        # and the report wrote the raw one.
        try:
            titles[canonicalize_url(raw_link)] = clean_title
        except ValueError:
            continue
    urls = [*titles, *_URL_RE.findall(report), *(citation_urls or [])]
    by_url: dict[str, SourceLead] = {}
    for raw_url in urls:
        try:
            url = canonicalize_url(raw_url)
        except ValueError:
            continue
        if url in by_url:
            if pass_number not in by_url[url].originating_passes:
                by_url[url].originating_passes.append(pass_number)
            continue
        host = urlsplit(url).hostname or ""
        identifiers = stated_identifiers(url)
        source_type = (
            "primary_or_authoritative"
            if identifiers
            or any(marker in host for marker in _AUTHORITATIVE_HOST_MARKERS)
            else "unknown"
        )
        title = titles.get(url) or titles.get(raw_url) or ""
        if title == _REDIRECTOR_HOST:
            title = ""
        snippet = ""
        for line in report.splitlines():
            if raw_url not in line and url not in line:
                continue
            if not title:
                inline = _MARKDOWN_LINK_RE.search(line)
                if inline and inline.group(1).strip() != _REDIRECTOR_HOST:
                    title = inline.group(1).strip()
            unlinked = _MARKDOWN_LINK_RE.sub(r"\1", line)
            prose = _LIST_MARKER_RE.sub("", unlinked).strip()
            if len(prose) > 15:
                snippet = prose
                if not title and len(prose.split(".")) > 1:
                    title = prose.split(".")[0].strip()
                break
        # No invented title and no invented summary. A lead labelled "Deep
        # Research Scholarly Evidence (pubs.acs.org)" disagrees with whatever
        # Crossref has registered for the DOI, and the verifier then reports a
        # real paper as "the registry record is a different document". An empty
        # title contradicts no registry, so the registered one is adopted.
        by_url[url] = SourceLead(
            canonical_url=url,
            title=title,
            summary=snippet[:180],
            identifiers=identifiers,
            source_type=source_type,
            originating_passes=[pass_number],
            raw_artifact_reference=raw_reference,
        )
    return list(by_url.values())


SUMMARY_CHARACTER_LIMIT = 12000
"""How much of one pass's report the manifest keeps for the Knowledge Base."""


def _report_summary(text: str) -> tuple[str, bool]:
    """The report as the dossier will print it, cut where a reader can see a cut.

    The limit used to be applied as a bare slice, which lands mid-word: a live
    run's Knowledge Base ended on "eliminating ambient thermal fl" and read as a
    sentence the provider had written that way. Cutting on a boundary and saying
    the cut happened are both this function's job, because the dossier is given
    the text and cannot tell a truncated report from a short one.
    """
    text = text.strip()
    if len(text) <= SUMMARY_CHARACTER_LIMIT:
        return text, False
    head = text[:SUMMARY_CHARACTER_LIMIT]
    for boundary, keep in (("\n\n", 0), (". ", 1), ("\n", 0)):
        cut = head.rfind(boundary)
        if cut > SUMMARY_CHARACTER_LIMIT // 2:
            return head[: cut + keep].strip(), True
    return head.rstrip(), True


def _fallback_narrative(
    question: str,
    report: str,
    pass_number: int,
    citation_urls: list[str] | None = None,
) -> DiscoveryNarrative:
    cited = citation_urls or []
    statements: list[DiscoveryStatement] = []
    for paragraph in re.split(r"\n\s*\n", report):
        urls = []
        for raw_url in _URL_RE.findall(paragraph):
            try:
                urls.append(canonicalize_url(raw_url))
            except ValueError:
                continue
        # A report that cites by marker spells out no URL at all, so without
        # this the fallback returns nothing from a pass that cited forty papers.
        # The marker is resolved positionally against the provider's own list,
        # which is the order the provider numbered them in.
        for marker in _CITATION_MARKER_RE.findall(paragraph):
            for number in marker.split(","):
                index = int(number) - 1
                if 0 <= index < len(cited):
                    urls.append(cited[index])
        if not urls:
            continue
        lowered = paragraph.lower()
        facet = next(
            (
                name
                for name, keywords in _FACET_KEYWORDS.items()
                if any(keyword in lowered for keyword in keywords)
            ),
            "supporting",
        )
        statements.append(
            DiscoveryStatement(
                text=re.sub(r"\s+", " ", paragraph).strip()[:4000],
                facet=facet,
                source_urls=list(dict.fromkeys(urls)),
                originating_pass=pass_number,
                relation="contradicts" if facet == "contradictory" else "neutral",
            )
        )
    return DiscoveryNarrative(
        question=question,
        research_directions=[question],
        statements=statements,
        summary=report,
        # This is a diagnostic about the normalizer, not a question the field has
        # left open, and Open Questions is where the report prints uncertainties.
        # A reader asking what this run could not settle was handed "No
        # citation-linked statements could be normalized." among seven real ones.
        uncertainties=[],
    )


def normalize_report(
    *,
    question: str,
    report: str,
    pass_number: int,
    normalizer: Callable[[str], str] | None = None,
    citation_urls: list[str] | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> DiscoveryNarrative:
    """Normalize a report, accepting only citations the provider itself returned.

    ``citation_urls`` is the provider's own citation metadata for this pass. It
    belongs here because a Deep Research report generally cites by number and
    carries the URLs alongside the prose rather than inside it, and the guard
    below -- which exists to stop a normalizer inventing a source -- was reading
    the report text as the whole of what the provider said. On a live wave that
    threw away every statement in all eight passes.
    """
    cited = []
    for raw_url in citation_urls or []:
        try:
            cited.append(canonicalize_url(raw_url))
        except ValueError:
            continue
    cited = list(dict.fromkeys(cited))
    narrative = None
    if normalizer is not None:
        # The numbered list is what makes "copied verbatim" achievable when the
        # report cites by marker: without it the only URLs a normalizer can copy
        # are the ones that happen to be spelled out in the prose.
        sources_block = (
            "\n\nSources cited by this report, in order:\n"
            + "\n".join(f"[{index}] {url}" for index, url in enumerate(cited, start=1))
            if cited
            else ""
        )
        # Verbatim is asked for of the URLs, and was taken to cover the prose: this
        # extractor is where "Die Lebensdauer der Elektrodenmaterialien wird durch
        # die Beschichtung stark erhöht" entered the findings of an English report,
        # lifted whole out of the German paper the pass cited.
        working_language = "\n".join(
            part
            for part in (language_clause(language), source_language_rule(language))
            if part
        )
        prompt = (
            "Extract one DiscoveryNarrative JSON object from the report below. "
            "Do not add facts or URLs. Each statement must contain its originating "
            f"pass {pass_number}, an evidence facet, and only source URLs copied "
            "verbatim from the report or from the numbered source list beneath "
            "it. Where the report cites a marker such as [3], resolve it against "
            "that list.\n\n"
            f"{working_language}\n\n"
            f"Research question: {question}\n\nReport:\n{report[:180000]}"
            f"{sources_block}"
        )
        try:
            raw = normalizer(prompt)
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                narrative = DiscoveryNarrative.model_validate_json(raw[start : end + 1])
        except (ValueError, ValidationError, json.JSONDecodeError):
            narrative = None
    if narrative is None:
        narrative = _fallback_narrative(question, report, pass_number, cited)

    report_urls = set(cited)
    for raw_url in _URL_RE.findall(report):
        try:
            report_urls.add(canonicalize_url(raw_url))
        except ValueError:
            continue
    accepted_statements = []
    for statement in narrative.statements:
        valid_urls = []
        for raw_url in statement.source_urls:
            try:
                url = canonicalize_url(raw_url)
            except ValueError:
                continue
            if url in report_urls:
                valid_urls.append(url)
        if valid_urls:
            statement.source_urls = list(dict.fromkeys(valid_urls))
            statement.originating_pass = pass_number
            accepted_statements.append(statement)
    narrative.statements = accepted_statements
    narrative.question = question
    # Recorded rather than inferred from the order the reports are printed in, which
    # skips any pass that came back with nothing to print.
    narrative.pass_number = pass_number
    # Applied to whichever summary survived -- the normalizer's or the fallback's
    # -- so the dossier is handed one kind of text however this pass was parsed.
    narrative.summary, narrative.truncated = _report_summary(
        narrative.summary or report
    )
    return narrative


def stated_identifiers(url: str) -> dict[str, str]:
    """The identifiers a locator states outright, read off the locator itself.

    Discovery reads these when it first writes a lead, and whatever replaces a
    lead's locator afterwards has to read them again. A redirector resolved to
    ``https://doi.org/10.1039/d5ta02510a`` carried the URL and not the DOI, so
    the copy of that paper another pass had found by its DOI stayed a second
    lead: same document, same address, two rows -- see ``lead_identity``, which
    matches on the DOI where there is one and on the address where there is not.
    """
    match = _DOI_RE.search(url)
    return {"doi": match.group(0).rstrip(".").lower()} if match else {}


def lead_identity(lead: SourceLead) -> str:
    """What makes two search results the same document.

    The canonical URL alone is not it. A grounding redirector mints one opaque
    token per citation, and the ones that cannot be followed keep that token --
    so a live report printed references 5, 6 and 7 as three separate papers when
    they were one paper found three times, under a title the three shared word
    for word. A DOI is the same document by definition however it was reached,
    and a redirector that resolved is matched on where it resolved to.
    """
    doi = (lead.identifiers.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = " ".join(lead.title.split()).casefold()
    if GROUNDING_REDIRECT_MARKER in lead.canonical_url and title:
        return f"title:{title}"
    return lead.canonical_url


def merge_leads(
    existing: list[SourceLead], additions: list[SourceLead]
) -> list[SourceLead]:
    merged = {lead_identity(lead): lead.model_copy(deep=True) for lead in existing}
    for lead in additions:
        current = merged.get(lead_identity(lead))
        if current is None:
            merged[lead_identity(lead)] = lead.model_copy(deep=True)
            continue
        # A redirector says which search found the paper; the resolved link says
        # which paper. Whichever copy carries the second one is the locator to
        # keep, whichever of them the merge happened to meet first.
        if (
            GROUNDING_REDIRECT_MARKER in current.canonical_url
            and GROUNDING_REDIRECT_MARKER not in lead.canonical_url
        ):
            current.canonical_url = lead.canonical_url
        current.originating_passes = list(
            dict.fromkeys([*current.originating_passes, *lead.originating_passes])
        )
        current.originating_statement_ids = list(
            dict.fromkeys(
                [
                    *current.originating_statement_ids,
                    *lead.originating_statement_ids,
                ]
            )
        )
        current.identifiers.update(lead.identifiers)
        # A paper found by both the replication search and the contradictory
        # search belongs under both, and dropping the second is how a facet
        # ends up looking empty when it was covered.
        current.facets = list(dict.fromkeys([*current.facets, *lead.facets]))
        current.claim_relations = list(
            dict.fromkeys([*current.claim_relations, *lead.claim_relations])
        )
        if not current.title and lead.title:
            current.title = lead.title
        if current.source_type == "unknown":
            current.source_type = lead.source_type
        # The one copy that was checked settles it for the document. Keeping the
        # first copy's "discovered_unverified" would throw away a retrieval that
        # already happened and quarantine a source this run has read.
        if (
            current.verification_status == "discovered_unverified"
            and lead.verification_status != "discovered_unverified"
        ):
            current.verification_status = lead.verification_status
            current.verification_note = lead.verification_note
    return list(merged.values())


def retain_leads(leads: list[SourceLead], limit: int) -> list[SourceLead]:
    """Cut a corpus down to ``limit`` leads without losing whole facets.

    This was ``leads[:limit]``, and the list it sliced is in the order the passes
    were ingested. A seven-facet fan-out returns its broadest pass first, so a
    live run that found three hundred and sixty-three leads kept ninety, all of
    them the supporting pass's, and threw away every lead the contradictory,
    negative-null, replication, methods, safety and retraction passes had
    returned -- twenty-one dollars of research, six facets of it discarded by a
    slice. The panel then reported one facet covered and the evidence floor
    failed, on a corpus that had covered all seven an hour earlier.

    Taking one lead per facet per turn spends the same budget and keeps the shape
    of what was searched. Untagged leads go last within each turn, not last
    overall: on an undecomposed pass they are all there is.
    """
    if len(leads) <= limit:
        return list(leads)
    queues: dict[str, list[SourceLead]] = {facet: [] for facet in (*FAN_OUT_FACETS, "")}
    for lead in leads:
        facet = next((item for item in lead.facets if item in queues), "")
        queues[facet].append(lead)
    kept: list[SourceLead] = []
    while len(kept) < limit and any(queues.values()):
        for queue in queues.values():
            if not queue:
                continue
            kept.append(queue.pop(0))
            if len(kept) >= limit:
                break
    return kept


def audit_coverage(
    narrative: DiscoveryNarrative,
    leads: list[SourceLead],
    previous: DiscoveryCoverage | None = None,
    *,
    searched_facets: set[str] | None = None,
) -> DiscoveryCoverage:
    """Score what the literature actually covers, and name what it does not.

    ``searched_facets`` is which facets a search was aimed at, which the caller
    knows and the report cannot say: a facet whose dedicated pass came back with
    "no such literature exists" leaves no statement behind to be counted, so
    without it that facet is indistinguishable from one nobody asked about.
    """
    facet_scores = {}
    statement_text = "\n".join(
        statement.text.lower() for statement in narrative.statements
    )
    # When the searches were decomposed by facet, the facet a statement carries
    # is a record of which search returned it, and guessing from its wording can
    # only make coverage look better than it was. A run whose contradictory
    # search came back empty scored that facet 1.0 because some other statement
    # happened to contain the word "inconsistent". The keyword pass is kept for
    # the undecomposed single-report case, where a tag is all the report's own
    # prose can supply.
    #
    # A tag alone is not coverage, or the fan-out would score itself: seven
    # passes go out tagged with seven facets, and every one of them comes back
    # tagged whether it found literature or reported that there is none. What
    # counts is a tagged statement that cites something, which is exactly the
    # difference between "we looked here" and "there is evidence here".
    searched = searched_facets or {
        statement.facet
        for statement in narrative.statements
        if statement.facet in EVIDENCE_FACETS
    }
    #
    # A lead counts for the same reason a cited statement does, and it has to:
    # tying coverage to prose alone made the score depend on how the provider
    # chose to cite. A live wave returned ninety citable papers through the
    # provider's own citation metadata rather than as links in the report text,
    # every statement was dropped for having no URL the report spelled out, and
    # the stage then reported 0% coverage and told the reader that all seven
    # passes "returned no citable source" while seven of them had returned
    # dozens. A lead is tagged with the facet of the pass that returned it, so
    # counting it says the same thing about the same act.
    tagged_facets = {
        statement.facet
        for statement in narrative.statements
        if statement.facet in EVIDENCE_FACETS and statement.source_urls
    }
    tagged_facets.update(
        facet for lead in leads for facet in lead.facets if facet in EVIDENCE_FACETS
    )
    for facet in EVIDENCE_FACETS:
        if searched:
            facet_scores[facet] = 1.0 if facet in tagged_facets else 0.0
            continue
        keyword_match = any(
            keyword in statement_text for keyword in _FACET_KEYWORDS[facet]
        )
        facet_scores[facet] = 1.0 if keyword_match else 0.0

    directions = narrative.research_directions or [narrative.question]
    direction_scores = {
        direction: (
            1.0
            if any(
                direction.lower()[:40] in statement.text.lower()
                for statement in narrative.statements
            )
            else min(1.0, len(narrative.statements) / max(1, len(directions) * 2))
        )
        for direction in directions
    }
    weighted_score = (
        sum(facet_scores.values()) + sum(direction_scores.values())
    ) / max(1, len(facet_scores) + len(direction_scores))
    authoritative = sum(
        lead.source_type == "primary_or_authoritative" for lead in leads
    )
    previous_authoritative = previous.authoritative_source_count if previous else 0
    gaps = [
        ResearchGap(
            direction="Evidence landscape",
            facet=facet,
            description=(
                # A facet that got its own pass and came back empty is a
                # different finding from one nothing ever asked about, and the
                # gap-closing pass is aimed by these descriptions.
                f"A pass dedicated to {FACET_PHRASES.get(facet, facet.replace('_', ' '))} "
                "returned no citable source."
                if facet in searched
                else "The discovery pass found no adequate "
                f"{FACET_PHRASES.get(facet, facet.replace('_', ' '))}."
            ),
            decision_impact=(
                "high"
                if facet in {"contradictory", "negative_null", "methods"}
                else "medium"
            ),
            priority=4 if facet in {"contradictory", "negative_null"} else 3,
        )
        for facet, score in facet_scores.items()
        if score == 0
    ]
    return DiscoveryCoverage(
        direction_scores=direction_scores,
        facet_scores=facet_scores,
        weighted_score=round(weighted_score, 4),
        sufficient=weighted_score >= SUFFICIENT_COVERAGE and not gaps,
        authoritative_source_count=authoritative,
        new_authoritative_source_count=max(0, authoritative - previous_authoritative),
        material_gaps_closed=max(0, len(previous.gaps) - len(gaps) if previous else 0),
        gaps=gaps,
    )


def _combined_narrative(
    question: str, narratives: list[DiscoveryNarrative]
) -> DiscoveryNarrative:
    """Everything discovered so far, as one narrative to score coverage against.

    Seven facet passes produce seven narratives that each answer a different
    question, and coverage is a property of the whole literature rather than of
    any one of them. Statements are deduplicated on what they say and where they
    say it came from, because two passes citing the same paper for the same
    finding is one finding.
    """
    seen: dict[tuple[str, tuple[str, ...], str, str], DiscoveryStatement] = {}
    for item in narratives:
        for statement in item.statements:
            seen.setdefault(
                (
                    statement.text,
                    tuple(statement.source_urls),
                    statement.facet,
                    statement.relation,
                ),
                statement,
            )
    return DiscoveryNarrative(
        question=question,
        research_directions=list(
            dict.fromkeys(
                direction
                for item in narratives
                for direction in item.research_directions
            )
        ),
        statements=list(seen.values()),
        disagreements=[value for item in narratives for value in item.disagreements],
        uncertainties=[value for item in narratives for value in item.uncertainties],
    )


def should_repeat(
    history: list[DiscoveryCoverage],
    estimated_cost_usd: float,
    *,
    max_passes: int = MAX_DEEP_RESEARCH_PASSES,
) -> tuple[bool, str]:
    if not history:
        return True, "initial_pass_required"
    latest = history[-1]
    if latest.sufficient:
        return False, "coverage_sufficient"
    if len(history) >= max_passes:
        return False, "maximum_passes_reached"
    if estimated_cost_usd >= DEFAULT_COST_LIMIT_USD:
        return False, "cost_limit_reached"
    if len(history) >= 2:
        improvement = latest.weighted_score - history[-2].weighted_score
        useful = (
            latest.new_authoritative_source_count > 0 or latest.material_gaps_closed > 0
        )
        if improvement < MIN_COVERAGE_IMPROVEMENT:
            return False, "coverage_improvement_below_threshold"
        if not useful:
            return False, "no_material_incremental_value"
    return True, "material_gaps_remain"


def build_research_prompt(
    session: Session,
    plan: ResearchPlan,
    *,
    pass_number: int,
    previous_manifest: DiscoveryManifest | None = None,
    facet: str = "",
) -> str:
    coverage_requirements = "\n".join(
        # The facet token is what a statement must be tagged with, so it is given
        # exactly as the contract spells it and glossed rather than prettified.
        f"- {facet}: {FACET_PHRASES[facet]}"
        for facet in EVIDENCE_FACETS
    )
    if facet:
        # One interaction per facet, all seven running at once. A single broad
        # report asked for the mechanism, the studies for and against it, the
        # replications, the null results and the retractions together, and
        # answered with the supporting literature -- which is what that question
        # deserves. Asked on its own, each facet is a search with its own answer.
        focus = (
            f"THIS PASS COVERS ONE FACET ONLY: {facet} — {FACET_PHRASES[facet]}.\n"
            "The other facets of this question are being researched in parallel "
            "by separate passes, so breadth outside your facet costs this run its "
            "coverage rather than adding to it. Go deep on yours: find the "
            "primary literature, name the studies, and tag every statement you "
            f"return with the facet {facet}.\n"
            "If the primary literature does not exist, say so plainly: finding "
            "nothing is a finding, and padding it out with adjacent material is a "
            "gap that has been hidden. Write that finding the way the report will "
            "print it -- name what you searched for, the constraint nothing met, "
            "and what the absence implies for the question. The words 'facet', "
            "'empty facet' and 'pass' describe how this run is organised and mean "
            # A live report printed "this constitutes a genuinely empty facet" and
            # "an honest empty facet" in the reader's own paragraphs: the prompt's
            # own phrasing, handed back as the finding.
            "nothing to a reader; the tag above is machine-read and belongs in the "
            "facet field of a statement, never in its prose."
        )
    elif pass_number == 1 or previous_manifest is None:
        focus = (
            "Build a broad, domain-appropriate evidence landscape. Identify distinct "
            "research directions, material disagreements, and evidence gaps."
        )
    else:
        gaps = previous_manifest.coverage_history[-1].gaps
        known = [lead.canonical_url for lead in previous_manifest.source_leads[:40]]
        focus = (
            "Research only these unresolved material gaps:\n"
            + "\n".join(f"- [{gap.facet}] {gap.description}" for gap in gaps[:12])
            + "\nAvoid duplicating these already discovered sources:\n"
            + "\n".join(f"- {url}" for url in known)
        )
    return (
        f"Research question: {session.question}\n"
        f"Research mode: {session.research_mode}\n"
        f"Intended claim: {plan.intended_claim}\n"
        f"Constraints: {'; '.join(plan.constraints) or 'none supplied'}\n"
        f"Literature-only fallback: {'yes' if session.literature_only else 'no'}\n\n"
        f"{focus}\n\nRequired evidence coverage:\n{coverage_requirements}\n\n"
        # The one prompt in the system that searches the open web, and the only
        # one that had no working language at all. A Chinese run's whole Knowledge
        # Base came back in English because this prompt never said otherwise.
        f"{session_language_clause(session)}"
        f"{source_language_rule(getattr(session, 'language', '') or DEFAULT_LANGUAGE)}"
        "\n\n"
        "Prefer primary sources, authoritative repositories, standards, registered "
        "studies, and datasets. Use readable citations with source URLs. Clearly "
        "separate findings, disagreement, inference, and proposals. If evidence is "
        "unavailable, say so rather than estimating or inventing it. Do not claim "
        "to have analyzed an unsupplied sequence, dataset, experiment, or result."
    )


@dataclass
class PlannedPass:
    """One Deep Research interaction the controller has decided to start."""

    pass_number: int
    facet: str
    prompt: str
    gap_ids: list[str]


@dataclass
class IterativeEvidenceDiscovery:
    transport: DeepResearchTransport
    artifact_store: EvidenceArtifactStore
    poll_interval_seconds: float = 15.0
    pass_timeout_seconds: float = DEFAULT_PASS_TIMEOUT_SECONDS
    cost_per_pass_usd: float = DEFAULT_COST_PER_PASS_USD
    max_passes: int = MAX_DEEP_RESEARCH_PASSES
    max_waves: int = 2
    """How many rounds of interactions a run may take, fan-out counting as one.

    Two: the facet fan-out, then one pass aimed at whatever it left open. The
    fan-out buys breadth and gives up gap-direction, because seven passes
    launched together cannot see what the others missed; the second wave is
    where that is bought back.
    """
    fan_out: bool = True
    registry_enricher: RegistryMetadataEnricher | None = None
    polls_per_invocation: int | None = None

    _TERMINAL = frozenset(
        {"completed", "failed", "cancelled", "incomplete", "budget_exceeded"}
    )
    _IN_FLIGHT = frozenset({"queued", "in_progress", "requires_action"})

    def _remaining_passes(self, manifest: DiscoveryManifest) -> int:
        """How many more interactions the pass and cost ceilings still allow."""
        by_count = self.max_passes - len(manifest.runs)
        if self.cost_per_pass_usd <= 0:
            return max(0, by_count)
        by_cost = int(
            (DEFAULT_COST_LIMIT_USD - manifest.estimated_cost_usd)
            // self.cost_per_pass_usd
        )
        return max(0, min(by_count, by_cost))

    def _plan_wave(
        self, session: Session, plan: ResearchPlan, manifest: DiscoveryManifest
    ) -> tuple[list[PlannedPass], str]:
        """The next round of interactions to start, or why there is not one."""
        budget = self._remaining_passes(manifest)
        if not budget:
            return [], (
                "maximum_passes_reached"
                if len(manifest.runs) >= self.max_passes
                else "cost_limit_reached"
            )
        started = len(manifest.runs)
        if not started:
            facets = list(FAN_OUT_FACETS) if self.fan_out else [""]
            if len(facets) > budget:
                # Never silently narrow the fan-out: a run that could only afford
                # four of the seven facets must say which three were never
                # searched, or its empty facets read as absent literature.
                dropped = facets[budget:]
                facets = facets[:budget]
                manifest.convergence_reason = "fan_out_truncated_by_budget:" + ",".join(
                    dropped
                )
            return [
                PlannedPass(
                    pass_number=index,
                    facet=facet,
                    prompt=build_research_prompt(
                        session, plan, pass_number=index, facet=facet
                    ),
                    gap_ids=[],
                )
                for index, facet in enumerate(facets, start=1)
            ], ""
        repeat, reason = should_repeat(
            manifest.coverage_history,
            manifest.estimated_cost_usd,
            max_passes=self.max_waves,
        )
        if not repeat:
            return [], reason
        gaps = manifest.coverage_history[-1].gaps if manifest.coverage_history else []
        return [
            PlannedPass(
                pass_number=started + 1,
                facet="",
                prompt=build_research_prompt(
                    session,
                    plan,
                    pass_number=started + 1,
                    previous_manifest=manifest,
                ),
                gap_ids=[gap.id for gap in gaps],
            )
        ], ""

    def _start_wave(
        self,
        session: Session,
        planned: list[PlannedPass],
        manifest: DiscoveryManifest,
    ) -> tuple[list[DeepResearchRun], dict[str, dict]]:
        """Dispatch a planned wave, recording each interaction as it is accepted.

        A pass that will not start is recorded as failed and the wave continues.
        Six facets researched and one refused is a run with a stated hole in it;
        raising here would throw away six interactions that have already been
        paid for.

        The payload ``start`` returned travels back with the runs, because a
        Deep Research interaction can already be complete when it is created and
        Vertex will refuse to be polled for one. Keeping only its id and status
        threw the report away and recorded the pass as "completed without a
        report" -- the whole wave, every time, for that shape of transport.
        """
        started: list[DeepResearchRun] = []
        payloads: dict[str, dict] = {}
        for item in planned:
            try:
                created = self.transport.start(
                    prompt=item.prompt,
                    pass_number=item.pass_number,
                    session_id=session.id,
                )
                interaction_id = str(created.get("id") or "")
            except Exception as exc:
                created, interaction_id = {}, ""
                failure = f"{type(exc).__name__}: {exc}"
            else:
                failure = "" if interaction_id else "Deep Research returned no ID."
            run = DeepResearchRun(
                pass_number=item.pass_number,
                facet=item.facet,
                interaction_id=interaction_id,
                status=(
                    str(created.get("status") or "queued")
                    if interaction_id
                    else "failed"
                ),
                error=failure,
                prompt_gap_ids=item.gap_ids,
                estimated_cost_usd=self.cost_per_pass_usd if interaction_id else 0.0,
            )
            if not interaction_id:
                run.completed_at = _now()
            manifest.runs.append(run)
            if interaction_id:
                started.append(run)
                payloads[interaction_id] = created
        return started, payloads

    def _await_wave(
        self,
        wave: list[DeepResearchRun],
        manifest: DiscoveryManifest,
        *,
        seed: dict[str, dict] | None = None,
        status_callback: Callable[[DeepResearchRun], None] | None,
        manifest_callback: Callable[[DiscoveryManifest], None] | None,
    ) -> dict[str, dict] | None:
        """Poll every interaction in a wave until all are terminal.

        ``None`` means the caller should return the manifest as it stands: the
        step-mode budget for this invocation is spent and a Cloud Task will resume,
        so nothing may be folded in yet or the resumed invocation would fold the
        same passes in twice.

        A wall-clock deadline is the other ending, and it returns the payloads it
        has. The passes that finished before it are paid for and complete, and the
        caller reads them before it stops.
        """
        payloads: dict[str, dict] = {
            run.interaction_id: (seed or {}).get(run.interaction_id)
            or {"id": run.interaction_id, "status": run.status}
            for run in wave
        }
        polls_this_invocation = 0
        if self.polls_per_invocation == 0 and any(
            run.status not in self._TERMINAL for run in wave
        ):
            manifest.convergence_reason = "interaction_in_progress"
            return None
        deadline = time.monotonic() + self.pass_timeout_seconds
        while any(
            str(payloads[run.interaction_id].get("status")) not in self._TERMINAL
            for run in wave
        ):
            if time.monotonic() >= deadline:
                for run in wave:
                    if (
                        str(payloads[run.interaction_id].get("status"))
                        in self._TERMINAL
                    ):
                        continue
                    run.status = "timed_out"
                    run.error = "Deep Research exceeded the local pass deadline."
                    run.completed_at = _now()
                    # Marked in the payload as well as on the run, because the
                    # ingest reads each pass's status back off its payload and
                    # would otherwise restore the last "in_progress" it polled.
                    payloads[run.interaction_id] = {
                        **payloads[run.interaction_id],
                        "status": "timed_out",
                    }
                manifest.convergence_reason = "deep_research_timed_out"
                return payloads
            time.sleep(self.poll_interval_seconds)
            polls_this_invocation += 1
            for run in wave:
                if str(payloads[run.interaction_id].get("status")) in self._TERMINAL:
                    continue
                payloads[run.interaction_id] = self.transport.get(run.interaction_id)
                run.poll_count += 1
                run.status = str(
                    payloads[run.interaction_id].get("status") or "in_progress"
                )
                if status_callback:
                    status_callback(run)
            if manifest_callback:
                manifest_callback(manifest)
            if (
                self.polls_per_invocation is not None
                and polls_this_invocation >= self.polls_per_invocation
                and any(run.status not in self._TERMINAL for run in wave)
            ):
                manifest.convergence_reason = "interaction_in_progress"
                return None
        return payloads

    def _ingest_wave(
        self,
        session: Session,
        wave: list[DeepResearchRun],
        payloads: dict[str, dict],
        manifest: DiscoveryManifest,
        *,
        normalizer: Callable[[str], str] | None,
    ) -> bool:
        """Fold a completed wave into the manifest and score coverage once.

        Coverage is scored per wave rather than per pass, because seven passes
        launched together are one observation of the literature: scoring each in
        turn would report six increments that no search caused.

        Returns whether the run may continue. It may not once every pass in a
        wave has failed, which is a transport problem rather than a thin
        literature and will not be improved by paying for another wave.
        """
        completed = 0
        for run in wave:
            payload = payloads.get(run.interaction_id) or {}
            run.status = str(payload.get("status") or run.status)
            run.completed_at = _now()
            run.usage = payload.get("usage") or {}
            run.raw_artifact_reference = self.artifact_store.put(
                session.id, run.pass_number, payload
            )
            manifest.estimated_cost_usd = round(
                manifest.estimated_cost_usd + self.cost_per_pass_usd, 2
            )
            if run.status != "completed":
                # Kept where the poller already said what went wrong in words. A pass
                # abandoned at the deadline carries no error of its own in its payload,
                # so serializing that payload would replace the sentence with "{}".
                run.error = run.error or json.dumps(
                    payload.get("error") or payload.get("incomplete_details") or {}
                )
                continue
            report = _extract_report(payload)
            if not report.strip():
                run.status = "failed"
                run.error = "Deep Research completed without a report."
                continue
            completed += 1
            steps = payload.get("steps") or []
            citation_urls = _payload_urls(steps)
            narrative = normalize_report(
                question=session.question,
                report=report,
                pass_number=run.pass_number,
                normalizer=normalizer,
                citation_urls=citation_urls,
                language=getattr(session, "language", "") or DEFAULT_LANGUAGE,
            )
            # What the pass was sent to cover, which for the gap-closing pass is
            # nothing in particular: it is planned with no facet because it covers
            # whatever the fan-out left open. The normalizer fills the field in from
            # the report where the plan left it empty, and one live gap pass came
            # back labelled with a facet name this run does not score -- a plan
            # nothing planned, invented after the fact by a model reading the prose.
            narrative.facet = run.facet
            if run.facet:
                # The pass was sent to cover one facet, so that is what its
                # statements are, whatever the keyword heuristic would have
                # guessed from their wording. This is the whole reason a
                # fan-out can score coverage honestly and a single broad report
                # cannot. The gap pass covers several, so its statements keep the
                # facets they were read with.
                for statement in narrative.statements:
                    statement.facet = run.facet
            additions = _source_leads(
                report,
                run.pass_number,
                run.raw_artifact_reference,
                citation_urls=citation_urls,
                citation_titles=_citation_titles(steps),
            )
            statement_ids_by_url: dict[str, list[str]] = {}
            for statement in narrative.statements:
                for url in statement.source_urls:
                    statement_ids_by_url.setdefault(url, []).append(statement.id)
            for lead in additions:
                lead.originating_statement_ids = statement_ids_by_url.get(
                    lead.canonical_url, []
                )
                if run.facet:
                    lead.facets = [run.facet]
            manifest.narratives.append(narrative)
            manifest.source_leads = merge_leads(manifest.source_leads, additions)
        if not completed:
            manifest.convergence_reason = (
                f"deep_research_{wave[0].status}" if wave else "deep_research_failed"
            )
            return False
        manifest.coverage_history.append(
            audit_coverage(
                _combined_narrative(session.question, manifest.narratives),
                manifest.source_leads,
                manifest.coverage_history[-1] if manifest.coverage_history else None,
                searched_facets={
                    run.facet
                    for run in manifest.runs
                    if run.facet in EVIDENCE_FACETS and run.status == "completed"
                },
            )
        )
        return True

    def run(
        self,
        session: Session,
        plan: ResearchPlan,
        *,
        manifest: DiscoveryManifest | None = None,
        normalizer: Callable[[str], str] | None = None,
        status_callback: Callable[[DeepResearchRun], None] | None = None,
        manifest_callback: Callable[[DiscoveryManifest], None] | None = None,
    ) -> DiscoveryManifest:
        manifest = manifest or DiscoveryManifest(question=session.question)
        seed: dict[str, dict] = {}
        while True:
            # An interaction already in flight is resumed rather than restarted.
            # This is what lets the Cloud Tasks worker step a fan-out: it returns
            # after one poll, and the next invocation picks up all seven.
            wave = [
                run
                for run in manifest.runs
                if run.status in self._IN_FLIGHT and run.interaction_id
            ]
            if not wave:
                planned, reason = self._plan_wave(session, plan, manifest)
                if not planned:
                    if not manifest.convergence_reason.startswith("fan_out_truncated"):
                        manifest.convergence_reason = reason
                    break
                wave, seed = self._start_wave(session, planned, manifest)
                if not wave:
                    manifest.convergence_reason = "deep_research_start_failed"
                    break
            if status_callback:
                for run in wave:
                    status_callback(run)
            if manifest_callback:
                manifest_callback(manifest)
            payloads = self._await_wave(
                wave,
                manifest,
                seed=seed,
                status_callback=status_callback,
                manifest_callback=manifest_callback,
            )
            seed = {}
            if payloads is None:
                return manifest
            # A deadline stops the search, but only after the wave it already paid for
            # is read. On a live run six of seven passes had come back when the seventh
            # ran long, and returning here unread discarded six completed Deep Research
            # reports: the stage reported seven passes, zero source leads and a floor
            # met by nothing, having bought the literature and thrown it away.
            timed_out = manifest.convergence_reason == "deep_research_timed_out"
            if not self._ingest_wave(
                session, wave, payloads, manifest, normalizer=normalizer
            ):
                break
            if manifest_callback:
                manifest_callback(manifest)
            if timed_out:
                break

        if manifest.coverage_history and not manifest.coverage_history[-1].sufficient:
            manifest.enrichment_requests = [
                EnrichmentRequest(
                    provider="google_search",
                    gap_ids=[gap.id],
                    query=f"{session.question} {gap.description}",
                    status="queued",
                )
                for gap in manifest.coverage_history[-1].gaps[:MAX_ENRICHMENT_REQUESTS]
            ]
        retained = retain_leads(manifest.source_leads, MAX_RETAINED_SOURCE_LEADS)
        manifest.leads_beyond_retention_ceiling = len(manifest.source_leads) - len(
            retained
        )
        if self.registry_enricher is not None:
            manifest.source_leads = self.registry_enricher.enrich(retained)
        else:
            manifest.source_leads = retained
        manifest.verification_handoff_source_ids = [
            lead.id for lead in manifest.source_leads
        ]
        if manifest_callback:
            manifest_callback(manifest)
        return manifest
