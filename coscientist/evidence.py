"""Iterative Deep Research discovery with deterministic provenance and gates."""

from __future__ import annotations

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

from .models import (
    EVIDENCE_FACETS,
    DeepResearchRun,
    DiscoveryCoverage,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EnrichmentRequest,
    ResearchGap,
    ResearchPlan,
    Session,
    SourceLead,
)

DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
MAX_DEEP_RESEARCH_PASSES = 3
MIN_COVERAGE_IMPROVEMENT = 0.05
SUFFICIENT_COVERAGE = 0.90
DEFAULT_COST_PER_PASS_USD = 3.0
DEFAULT_COST_WARNING_USD = 6.0
DEFAULT_COST_LIMIT_USD = 10.0
DEFAULT_PASS_TIMEOUT_SECONDS = 3900
MAX_ENRICHMENT_REQUESTS = 6
MAX_REGISTRY_REQUESTS = 30
MAX_RETAINED_SOURCE_LEADS = 90

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
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


class GeminiDeepResearchTransport:
    """Small Interactions API adapter kept separate for deterministic tests."""

    def __init__(self, api_key: str | None = None):
        from google import genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            self._client = genai.Client(api_key=resolved_key)
        else:
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            if not project_id:
                try:
                    import google.auth
                    _, default_project = google.auth.default()
                    project_id = default_project
                except Exception:
                    pass
            if project_id:
                self._client = genai.Client(
                    vertexai=True, project=project_id, location=location
                )
            else:
                self._client = genai.Client(vertexai=True)

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

    def __init__(self):
        from google import genai

        self._client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location="global",
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


def _extract_report(payload: dict) -> str:
    if output := payload.get("output_text"):
        return str(output)
    for step in reversed(payload.get("steps") or []):
        if not isinstance(step, dict):
            continue
        for key in ("content", "output", "result"):
            value = step.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                texts = [
                    str(item.get("text", ""))
                    for item in value
                    if isinstance(item, dict) and item.get("text")
                ]
                if texts:
                    return "\n".join(texts)
    return ""


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
) -> list[SourceLead]:
    titles = {url: title.strip() for title, url in _MARKDOWN_LINK_RE.findall(report)}
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
        identifiers = {}
        if match := _DOI_RE.search(url):
            identifiers["doi"] = match.group(0).rstrip(".").lower()
        source_type = (
            "primary_or_authoritative"
            if identifiers
            or any(marker in host for marker in _AUTHORITATIVE_HOST_MARKERS)
            else "unknown"
        )
        by_url[url] = SourceLead(
            canonical_url=url,
            title=titles.get(raw_url, host),
            identifiers=identifiers,
            source_type=source_type,
            originating_passes=[pass_number],
            raw_artifact_reference=raw_reference,
        )
    return list(by_url.values())


def _fallback_narrative(
    question: str, report: str, pass_number: int
) -> DiscoveryNarrative:
    statements: list[DiscoveryStatement] = []
    for paragraph in re.split(r"\n\s*\n", report):
        urls = []
        for raw_url in _URL_RE.findall(paragraph):
            try:
                urls.append(canonicalize_url(raw_url))
            except ValueError:
                continue
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
        summary=report[:12000],
        uncertainties=(
            [] if statements else ["No citation-linked statements could be normalized."]
        ),
    )


def normalize_report(
    *,
    question: str,
    report: str,
    pass_number: int,
    normalizer: Callable[[str], str] | None = None,
) -> DiscoveryNarrative:
    """Normalize a report, accepting only citations present in the source report."""
    narrative = None
    if normalizer is not None:
        prompt = (
            "Extract one DiscoveryNarrative JSON object from the report below. "
            "Do not add facts or URLs. Each statement must contain its originating "
            f"pass {pass_number}, an evidence facet, and only source URLs copied "
            "verbatim from the report.\n\n"
            f"Research question: {question}\n\nReport:\n{report[:180000]}"
        )
        try:
            raw = normalizer(prompt)
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                narrative = DiscoveryNarrative.model_validate_json(raw[start : end + 1])
        except (ValueError, ValidationError, json.JSONDecodeError):
            narrative = None
    if narrative is None:
        narrative = _fallback_narrative(question, report, pass_number)

    report_urls = set()
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
    return narrative


def merge_leads(
    existing: list[SourceLead], additions: list[SourceLead]
) -> list[SourceLead]:
    merged = {lead.canonical_url: lead.model_copy(deep=True) for lead in existing}
    for lead in additions:
        current = merged.get(lead.canonical_url)
        if current is None:
            merged[lead.canonical_url] = lead.model_copy(deep=True)
            continue
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
        if not current.title and lead.title:
            current.title = lead.title
        if current.source_type == "unknown":
            current.source_type = lead.source_type
    return list(merged.values())


def audit_coverage(
    narrative: DiscoveryNarrative,
    leads: list[SourceLead],
    previous: DiscoveryCoverage | None = None,
) -> DiscoveryCoverage:
    facet_scores = {}
    statement_text = "\n".join(
        statement.text.lower() for statement in narrative.statements
    )
    for facet in EVIDENCE_FACETS:
        tagged = any(statement.facet == facet for statement in narrative.statements)
        keyword_match = any(
            keyword in statement_text for keyword in _FACET_KEYWORDS[facet]
        )
        facet_scores[facet] = 1.0 if tagged or keyword_match else 0.0

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
            description=f"No adequate {facet.replace('_', ' ')} evidence was discovered.",
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
) -> str:
    coverage_requirements = "\n".join(
        f"- {facet.replace('_', ' ')}" for facet in EVIDENCE_FACETS
    )
    if pass_number == 1 or previous_manifest is None:
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
        "Prefer primary sources, authoritative repositories, standards, registered "
        "studies, and datasets. Use readable citations with source URLs. Clearly "
        "separate findings, disagreement, inference, and proposals. If evidence is "
        "unavailable, say so rather than estimating or inventing it. Do not claim "
        "to have analyzed an unsupplied sequence, dataset, experiment, or result."
    )


@dataclass
class IterativeEvidenceDiscovery:
    transport: DeepResearchTransport
    artifact_store: EvidenceArtifactStore
    poll_interval_seconds: float = 15.0
    pass_timeout_seconds: float = DEFAULT_PASS_TIMEOUT_SECONDS
    cost_per_pass_usd: float = DEFAULT_COST_PER_PASS_USD
    max_passes: int = MAX_DEEP_RESEARCH_PASSES
    registry_enricher: RegistryMetadataEnricher | None = None
    polls_per_invocation: int | None = None

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
        while True:
            repeat, reason = should_repeat(
                manifest.coverage_history,
                manifest.estimated_cost_usd,
                max_passes=self.max_passes,
            )
            if not repeat:
                manifest.convergence_reason = reason
                break
            resumable = next(
                (
                    item
                    for item in reversed(manifest.runs)
                    if item.status in {"queued", "in_progress", "requires_action"}
                    and item.interaction_id
                ),
                None,
            )
            pass_number = resumable.pass_number if resumable else len(manifest.runs) + 1
            prompt = build_research_prompt(
                session,
                plan,
                pass_number=pass_number,
                previous_manifest=manifest if manifest.runs else None,
            )
            active = resumable
            if active is None:
                created = self.transport.start(
                    prompt=prompt,
                    pass_number=pass_number,
                    session_id=session.id,
                )
                interaction_id = str(created.get("id") or "")
                if not interaction_id:
                    raise RuntimeError("Deep Research returned no interaction ID.")
                run = DeepResearchRun(
                    pass_number=pass_number,
                    interaction_id=interaction_id,
                    status=str(created.get("status") or "queued"),
                    prompt_gap_ids=[
                        gap.id
                        for gap in (
                            manifest.coverage_history[-1].gaps
                            if manifest.coverage_history
                            else []
                        )
                    ],
                    estimated_cost_usd=self.cost_per_pass_usd,
                )
                manifest.runs.append(run)
                payload = created
            else:
                run = active
                interaction_id = run.interaction_id
                payload = {"id": interaction_id, "status": run.status}
            if status_callback:
                status_callback(run)
            if manifest_callback:
                manifest_callback(manifest)
            polls_this_invocation = 0
            if self.polls_per_invocation == 0 and run.status != "completed":
                manifest.convergence_reason = "interaction_in_progress"
                return manifest

            deadline = time.monotonic() + self.pass_timeout_seconds
            while str(payload.get("status")) not in {
                "completed",
                "failed",
                "cancelled",
                "incomplete",
                "budget_exceeded",
            }:
                if time.monotonic() >= deadline:
                    run.status = "timed_out"
                    run.error = "Deep Research exceeded the local pass deadline."
                    run.completed_at = _now()
                    manifest.convergence_reason = "deep_research_timed_out"
                    return manifest
                time.sleep(self.poll_interval_seconds)
                payload = self.transport.get(interaction_id)
                polls_this_invocation += 1
                run.poll_count += 1
                run.status = str(payload.get("status") or "in_progress")
                if status_callback:
                    status_callback(run)
                if manifest_callback:
                    manifest_callback(manifest)
                if (
                    self.polls_per_invocation is not None
                    and polls_this_invocation >= self.polls_per_invocation
                    and run.status
                    not in {
                        "completed",
                        "failed",
                        "cancelled",
                        "incomplete",
                        "budget_exceeded",
                    }
                ):
                    manifest.convergence_reason = "interaction_in_progress"
                    return manifest

            run.status = str(payload.get("status"))
            run.completed_at = _now()
            run.usage = payload.get("usage") or {}
            run.raw_artifact_reference = self.artifact_store.put(
                session.id, pass_number, payload
            )
            manifest.estimated_cost_usd = round(
                manifest.estimated_cost_usd + self.cost_per_pass_usd, 2
            )
            if manifest_callback:
                manifest_callback(manifest)
            if run.status != "completed":
                run.error = json.dumps(
                    payload.get("error") or payload.get("incomplete_details") or {}
                )
                manifest.convergence_reason = f"deep_research_{run.status}"
                break

            report = _extract_report(payload)
            if not report.strip():
                run.status = "failed"
                run.error = "Deep Research completed without a report."
                manifest.convergence_reason = "empty_deep_research_report"
                break
            narrative = normalize_report(
                question=session.question,
                report=report,
                pass_number=pass_number,
                normalizer=normalizer,
            )
            additions = _source_leads(
                report,
                pass_number,
                run.raw_artifact_reference,
                citation_urls=_payload_urls(payload.get("steps") or []),
            )
            statement_ids_by_url: dict[str, list[str]] = {}
            for statement in narrative.statements:
                for url in statement.source_urls:
                    statement_ids_by_url.setdefault(url, []).append(statement.id)
            for lead in additions:
                lead.originating_statement_ids = statement_ids_by_url.get(
                    lead.canonical_url, []
                )
            manifest.narratives.append(narrative)
            manifest.source_leads = merge_leads(manifest.source_leads, additions)
            combined_statements = {}
            for item in manifest.narratives:
                for statement in item.statements:
                    key = (
                        statement.text,
                        tuple(statement.source_urls),
                        statement.facet,
                        statement.relation,
                    )
                    combined_statements.setdefault(key, statement)
            combined = DiscoveryNarrative(
                question=session.question,
                research_directions=list(
                    dict.fromkeys(
                        direction
                        for item in manifest.narratives
                        for direction in item.research_directions
                    )
                ),
                statements=list(combined_statements.values()),
                disagreements=[
                    value
                    for item in manifest.narratives
                    for value in item.disagreements
                ],
                uncertainties=[
                    value
                    for item in manifest.narratives
                    for value in item.uncertainties
                ],
            )
            coverage = audit_coverage(
                combined,
                manifest.source_leads,
                manifest.coverage_history[-1] if manifest.coverage_history else None,
            )
            manifest.coverage_history.append(coverage)
            if manifest_callback:
                manifest_callback(manifest)

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
        if self.registry_enricher is not None:
            manifest.source_leads = self.registry_enricher.enrich(
                manifest.source_leads[:MAX_RETAINED_SOURCE_LEADS]
            )
        else:
            manifest.source_leads = manifest.source_leads[:MAX_RETAINED_SOURCE_LEADS]
        manifest.verification_handoff_source_ids = [
            lead.id for lead in manifest.source_leads
        ]
        if manifest_callback:
            manifest_callback(manifest)
        return manifest
