"""Deterministic, human-readable views over validated research artifacts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .citations import (
    CandidateCitations,
    latest_evidence_packet,
    resolve_candidate,
)
from .evidence import evaluate_evidence_floor
from .models import (
    CREDITED_STATUSES,
    EVIDENCE_FACETS,
    FACET_PHRASES,
    Candidate,
    CandidatePopulation,
    DiscoveryManifest,
    DossierManifest,
    EvidenceClaim,
    EvidencePacket,
    EvolutionCycle,
    ResearchLandscape,
    ResearchPlan,
    ReviewSet,
    Session,
    SourceLead,
    SourceRecord,
    TournamentState,
)
from .narrative import _number_word
from .parity import DEFAULT_ELO, UNMEASURED_MOVEMENT

PRESENTATION_SCHEMA_VERSION = "1"


def _latest_payload(session: Session, schema_name: str) -> dict[str, Any] | None:
    return next(
        (
            artifact.payload
            for artifact in reversed(session.artifacts)
            if artifact.schema_name == schema_name and artifact.payload
        ),
        None,
    )


def _population(session: Session) -> CandidatePopulation | None:
    payload = _latest_payload(session, "CandidatePopulation")
    return CandidatePopulation.model_validate(payload) if payload else None


def _candidate_index(
    population: CandidatePopulation | None,
) -> tuple[dict[str, Candidate], dict[str, str]]:
    candidates = population.candidates if population else []
    return (
        {candidate.id: candidate for candidate in candidates},
        {
            candidate.id: f"Candidate {index}"
            for index, candidate in enumerate(candidates, 1)
        },
    )


def _candidate_card(
    candidate: Candidate,
    label: str,
    *,
    reviews: list[dict[str, Any]] | None = None,
    rank: int | None = None,
    elo: float | None = None,
    shortlisted: bool = False,
    citations: CandidateCitations | None = None,
) -> dict[str, Any]:
    fatal_flaws = [
        flaw for review in reviews or [] for flaw in review.get("fatal_flaws", [])
    ]
    return {
        "candidate_id": candidate.id,
        "label": label,
        "claim": candidate.claim,
        "rationale": candidate.rationale,
        "strategy": candidate.generation_strategy,
        "predictions": candidate.predictions,
        "alternatives": candidate.alternatives,
        "falsifier": candidate.falsifier,
        "dependencies": candidate.dependencies,
        "risks": candidate.risks,
        "go_no_go_tests": candidate.go_no_go_tests,
        "evidence_ids": candidate.evidence_ids,
        # A citation nobody can resolve is worse than no citation, so the
        # support verdict travels beside the raw ids everywhere they are shown.
        "evidence_support": citations.support if citations else "unknown",
        "unresolved_evidence_ids": citations.unresolved if citations else [],
        "reviews": reviews or [],
        "fatal_flaws": fatal_flaws,
        "rank": rank,
        "elo": round(elo, 1) if elo is not None else None,
        "shortlisted": shortlisted,
    }


def _reviews(session: Session) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in session.artifacts:
        if artifact.schema_name != "ReviewSet" or not artifact.payload:
            continue
        for review in ReviewSet.model_validate(artifact.payload).reviews:
            grouped[review.candidate_id].append(review.model_dump(mode="json"))
    return grouped


_STATUS_VIEW: dict[str, tuple[str, str, str]] = {
    "verified": (
        "Verified",
        "verified",
        "The document was retrieved and the cited passage was found in it.",
    ),
    "corrected": (
        "Verified against a correction",
        "verified",
        "The document was retrieved, and the record carries a published correction.",
    ),
    "metadata_verified": (
        "Registry-confirmed",
        "partial",
        "A registry confirms the paper exists and is the one cited, but the full "
        "text could not be read, so nothing has checked what it says.",
    ),
    "discovered_unverified": (
        "Unverified",
        "quarantined",
        "The search returned it and verification has not reached it.",
    ),
    "inaccessible": (
        "Unreachable",
        "quarantined",
        "Neither the document nor a registry record could be obtained.",
    ),
    "retracted": (
        "Retracted",
        "retracted",
        "A registry records a retraction. Nothing may rest on this source.",
    ),
}
"""How each verification status is named, toned and explained to a reader.

Verification status is the primary axis of this panel, so the vocabulary is
defined once, next to the meanings, rather than being guessed at by whichever
surface happens to render it.
"""

_RELATION_VIEW = {
    "supports": "Supports",
    "contradicts": "Contradicts",
    "neutral": "Background",
}


def _facet_heading(facet: str) -> str:
    phrase = FACET_PHRASES.get(facet, facet.replace("_", " "))
    return phrase[:1].upper() + phrase[1:]


def _citation_line(
    authors: list[str], year: int | None, container: str, identifiers: dict[str, str]
) -> str:
    """One line that names the work, the way a reader would cite it."""
    parts: list[str] = []
    if authors:
        parts.append(authors[0] if len(authors) == 1 else f"{authors[0]} et al.")
    if year:
        parts.append(str(year))
    if container:
        parts.append(container)
    doi = identifiers.get("doi", "")
    if doi:
        parts.append(f"doi:{doi}")
    return " · ".join(parts)


def _source_card(
    *,
    url: str,
    lead: SourceLead | None,
    record: SourceRecord | None,
    claims: list[EvidenceClaim],
) -> dict[str, Any]:
    """One source, described by what is known about it rather than by its URL.

    A title and a link is a reading list. What a researcher needs before they can
    use a source is what was attributed to it, whether that attribution was
    checked, and whether the source agrees or disagrees with where the run is
    heading -- so all three travel with every entry.
    """
    status = (
        record.verification_status
        if record is not None
        else (lead.verification_status if lead else "discovered_unverified")
    )
    label, tone, meaning = _STATUS_VIEW.get(
        status, (status.replace("_", " ").capitalize(), "quarantined", "")
    )
    note = (record.verification_note if record else "") or (
        lead.verification_note if lead else ""
    )
    authors = (record.authors if record and record.authors else None) or (
        lead.authors if lead else []
    )
    year = (record.year if record and record.year else None) or (
        lead.year if lead else None
    )
    identifiers = {
        **(lead.identifiers if lead else {}),
        **(record.identifiers if record else {}),
    }
    facets = list(
        dict.fromkeys(
            [*(lead.facets if lead else []), *([record.facet] if record else [])]
        )
    )
    relations = list(
        dict.fromkeys(
            [
                *(claim.relation for claim in claims),
                *(lead.claim_relations if lead else []),
            ]
        )
    )
    return {
        "url": url,
        "title": (lead.title if lead else "")
        or (record.title if record else "")
        or url,
        "source_type": (lead.source_type if lead else "")
        or (record.source_type if record else "unknown"),
        "provider": lead.provider if lead else "verification",
        "citation": _citation_line(
            authors, year, record.container if record else "", identifiers
        ),
        "status": status,
        "status_label": label,
        "status_tone": tone,
        "status_meaning": meaning,
        "verification_note": note,
        "facets": [facet for facet in facets if facet],
        "relations": [relation for relation in relations if relation],
        "claims": [
            {
                "text": claim.claim,
                "relation": claim.relation,
                "relation_label": _RELATION_VIEW.get(claim.relation, claim.relation),
                "location": claim.exact_location,
                "confidence": round(claim.confidence, 2),
                "limitations": claim.limitations,
            }
            for claim in claims
        ],
        "passes": lead.originating_passes if lead else [],
    }


def _source_cards(
    manifest: DiscoveryManifest, packet: EvidencePacket | None
) -> list[dict[str, Any]]:
    records = {source.url: source for source in packet.sources} if packet else {}
    claims: dict[str, list[EvidenceClaim]] = defaultdict(list)
    if packet:
        for claim in packet.claims:
            if claim.source_id:
                claims[claim.source_id].append(claim)
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lead in manifest.source_leads:
        record = records.get(lead.canonical_url)
        seen.add(lead.canonical_url)
        cards.append(
            _source_card(
                url=lead.canonical_url,
                lead=lead,
                record=record,
                claims=claims.get(record.id, []) if record else [],
            )
        )
    # A verified source the manifest never held is rare and is exactly the kind
    # of thing that should not vanish from the panel: it means the corpus and
    # the discovery record disagree, which a reader should be able to see.
    for source in packet.sources if packet else []:
        if source.url in seen:
            continue
        cards.append(
            _source_card(
                url=source.url,
                lead=None,
                record=source,
                claims=claims.get(source.id, []),
            )
        )
    return cards


def _floor_headline(floor: Any, credited: int) -> str:
    if floor.met:
        return (
            f"{_number_word(credited)} usable "
            f"{'source' if credited == 1 else 'sources'} across "
            f"{_number_word(len(floor.facets_covered)).lower()} "
            f"{'facet' if len(floor.facets_covered) == 1 else 'facets'} — the "
            "evidence floor is met"
        )
    if not credited:
        return "No source has been verified — the evidence floor is not met"
    return (
        f"{_number_word(credited)} usable "
        f"{'source' if credited == 1 else 'sources'}, "
        f"{_number_word(len(floor.facets_covered)).lower()} "
        f"{'facet' if len(floor.facets_covered) == 1 else 'facets'} covered — the "
        "evidence floor is not met"
    )


def _evidence_trust_view(
    manifest: DiscoveryManifest, packet: EvidencePacket | None
) -> dict[str, Any]:
    """The evidence panel, organised by how far each source can be trusted.

    The panel this replaces listed every lead's title and URL in one flat block,
    which told a reader nothing they could act on: forty-four rows, all marked
    unverified, with no indication of which claim any of them was found for or
    which of them disagreed with the rest. Grouping by facet says what kind of
    evidence the corpus has and, by leaving the empty facets visible, what kind
    it does not; the quarantine list keeps what cannot be relied on where it can
    still be read, with the reason it is there.
    """
    floor = evaluate_evidence_floor(packet, manifest) if packet else None
    cards = _source_cards(manifest, packet)
    credited = [card for card in cards if card["status"] in CREDITED_STATUSES]
    quarantined = [card for card in cards if card["status"] not in CREDITED_STATUSES]
    coverage = manifest.coverage_history[-1] if manifest.coverage_history else None
    gaps: dict[str, list[dict[str, str]]] = defaultdict(list)
    for gap in coverage.gaps if coverage else []:
        gaps[gap.facet].append(
            {"description": gap.description, "impact": gap.decision_impact}
        )

    facets = [
        {
            "facet": facet,
            "label": _facet_heading(facet),
            "score": round(
                (coverage.facet_scores.get(facet, 0.0) if coverage else 0.0) * 100
            ),
            "sources": [card for card in credited if facet in card["facets"]],
            "gaps": gaps.get(facet, []),
        }
        for facet in EVIDENCE_FACETS
    ]
    unattributed = [card for card in credited if not card["facets"]]
    if unattributed:
        facets.append(
            {
                "facet": "unattributed",
                "label": "Not attributed to a facet",
                "score": None,
                "sources": unattributed,
                "gaps": [],
            }
        )
    return {
        "headline": _floor_headline(floor, len(credited)) if floor else "",
        "verification_ran": packet is not None,
        "floor": floor.model_dump(mode="json") if floor else None,
        "floor_details": (
            [
                {
                    "label": "Weighted credit",
                    "value": f"{floor.weighted_credit:g} of "
                    f"{floor.required_credit:g} required",
                    "met": floor.credit_met,
                },
                {
                    "label": "Facets covered",
                    "value": f"{len(floor.facets_covered)} of "
                    f"{floor.required_facets} required",
                    "met": floor.facets_met,
                },
                {
                    "label": "Disconfirming evidence",
                    "value": (
                        f"{floor.disconfirming_sources} found"
                        if floor.disconfirming_sources
                        else (
                            "none found, and the run did look"
                            if floor.searched_for_disconfirming
                            else "none, and the run never searched for it"
                        )
                    ),
                    "met": bool(
                        floor.disconfirming_sources or floor.searched_for_disconfirming
                    ),
                },
            ]
            if floor
            else []
        ),
        "shortfalls": floor.shortfalls if floor else [],
        "facets": facets,
        "quarantine": quarantined,
        "legend": [
            {"status": status, "label": label, "tone": tone, "meaning": meaning}
            for status, (label, tone, meaning) in _STATUS_VIEW.items()
        ],
        "verified_count": sum(card["status_tone"] == "verified" for card in cards),
        "metadata_verified_count": sum(
            card["status"] == "metadata_verified" for card in cards
        ),
        "quarantined_count": len(quarantined),
    }


def _base(stage: str, kind: str, summary: str) -> dict[str, Any]:
    return {
        "schema_version": PRESENTATION_SCHEMA_VERSION,
        "stage": stage,
        "kind": kind,
        "summary": summary,
        "metrics": [],
        "candidates": [],
        "reviews": [],
        "ranking": [],
        "comparison_rounds": [],
        "evolution": [],
        "clusters": [],
        "recommendations": [],
        "evidence": None,
        "details": [],
        "technical_details_available": True,
    }


def build_stage_presentation(session: Session, stage: str) -> dict[str, Any] | None:
    """Build a compact display model without asking a model to reformat output."""
    if stage == "scope":
        payload = _latest_payload(session, "ResearchPlan")
        if not payload:
            return None
        plan = ResearchPlan.model_validate(payload)
        result = _base(stage, "scope", plan.intended_claim)
        result["details"] = [
            {"label": "Research question", "value": plan.question},
            {"label": "Research mode", "value": plan.research_mode},
            {"label": "Assumptions", "value": plan.assumptions},
            {"label": "Constraints", "value": plan.constraints},
            {"label": "Success criteria", "value": plan.success_criteria},
            {"label": "Stopping criteria", "value": plan.stopping_criteria},
            {
                "label": "Governance requirements",
                "value": plan.governance_requirements,
            },
        ]
        return result

    if stage == "evidence":
        payload = _latest_payload(session, "DiscoveryManifest")
        if not payload:
            return None
        manifest = DiscoveryManifest.model_validate(payload)
        latest = manifest.coverage_history[-1] if manifest.coverage_history else None
        providers = sorted({lead.provider for lead in manifest.source_leads})
        packet = latest_evidence_packet(session)
        trust = _evidence_trust_view(manifest, packet)
        result = _base(
            stage,
            "evidence",
            # Naming the wrong provider is worse than naming none: a reader who
            # sees "Deep Research" trusts the leads further than search hits earn.
            trust["headline"]
            or (
                f"Knowledge landscape from {', '.join(providers)} — discovered, "
                "not yet verified"
                if providers
                else "Knowledge landscape — nothing was discovered"
            ),
        )
        result["evidence"] = trust
        result["metrics"] = [
            {"label": "Deep Research passes", "value": len(manifest.runs)},
            {"label": "Source leads", "value": len(manifest.source_leads)},
            # Counting anything as quarantined before the verifier has run would
            # report the stage's own incompleteness as a finding about the
            # literature, so the trust counters appear once there is a packet.
            *(
                [
                    {"label": "Verified", "value": trust["verified_count"]},
                    {
                        "label": "Registry-confirmed",
                        "value": trust["metadata_verified_count"],
                    },
                    {"label": "Quarantined", "value": trust["quarantined_count"]},
                ]
                if packet
                else []
            ),
            {
                "label": "Coverage",
                "value": round(latest.weighted_score * 100, 1) if latest else 0,
                "unit": "%",
            },
            {
                "label": "Estimated cost",
                "value": manifest.estimated_cost_usd,
                "unit": "USD",
            },
        ]
        result["details"] = [
            {"label": "Discovery provider", "value": providers or ["none"]},
            {
                "label": "Passes",
                "value": [
                    {
                        "pass": run.pass_number,
                        "facet": run.facet or "gap-closing",
                        "status": run.status,
                        "elapsed": [run.started_at, run.completed_at],
                        "cost": run.estimated_cost_usd,
                        "error": run.error,
                    }
                    for run in manifest.runs
                ],
            },
            {"label": "Stop reason", "value": manifest.convergence_reason},
        ]
        if manifest.stored_interaction_notice:
            result["details"].insert(
                1,
                {
                    "label": "Stored interaction notice",
                    "value": (
                        "Deep Research uses stored Gemini interactions so "
                        "background research can complete asynchronously."
                    ),
                },
            )
        return result

    population = _population(session)
    by_id, labels = _candidate_index(population)
    reviews_by_id = _reviews(session)

    if stage == "generate":
        if population is None:
            return None
        result = _base(
            stage,
            "candidates",
            f"{len(population.candidates)} competing, falsifiable candidates",
        )
        result["metrics"] = [
            {"label": "Candidates", "value": len(population.candidates)},
            {
                "label": "Generation strategies",
                "value": len(
                    {
                        candidate.generation_strategy
                        for candidate in population.candidates
                    }
                ),
            },
        ]
        packet = latest_evidence_packet(session)
        result["candidates"] = [
            _candidate_card(
                candidate,
                labels[candidate.id],
                citations=resolve_candidate(candidate, packet),
            )
            for candidate in population.candidates
        ]
        evidence_payloads = [
            artifact.payload
            for artifact in session.artifacts
            if artifact.schema_name == "EvidencePacket" and artifact.payload
        ]
        evidence = [
            EvidencePacket.model_validate(payload) for payload in evidence_payloads
        ]
        result["details"] = [
            {
                "label": "Evidence status",
                "value": [
                    {
                        "sources": len(packet.sources),
                        "claims": len(packet.claims),
                        "verified": packet.verified,
                        "limitations": packet.limitations,
                    }
                    for packet in evidence
                ],
            },
            {
                "label": "Comparison criteria",
                "value": population.comparison_criteria,
            },
            {
                "label": "Diversity dimensions",
                "value": population.diversity_dimensions,
            },
        ]
        return result

    if stage == "reflect":
        if population is None:
            return None
        result = _base(
            stage,
            "reviews",
            "Independent evidence, novelty, methods, impact, and governance review",
        )
        packet = latest_evidence_packet(session)
        result["candidates"] = [
            _candidate_card(
                candidate,
                labels[candidate.id],
                reviews=reviews_by_id.get(candidate.id, []),
                citations=resolve_candidate(candidate, packet),
            )
            for candidate in population.candidates
        ]
        result["metrics"] = [
            {
                "label": "Reviews",
                "value": sum(len(items) for items in reviews_by_id.values()),
            },
            {
                "label": "Candidates with fatal flaws",
                "value": sum(
                    any(review["fatal_flaws"] for review in reviews_by_id.get(cid, []))
                    for cid in by_id
                ),
            },
        ]
        return result

    if stage == "rank":
        payload = _latest_payload(session, "TournamentState")
        if not payload or population is None:
            return None
        tournament = TournamentState.model_validate(payload)
        ordered = sorted(
            tournament.ratings,
            key=lambda candidate_id: tournament.ratings[candidate_id],
            reverse=True,
        )
        result = _base(
            stage,
            "ranking",
            f"Shortlist of {len(tournament.shortlist_ids)} candidates after tournament review",
        )
        result["ranking"] = [
            {
                "rank": index,
                "candidate_id": candidate_id,
                "label": labels.get(candidate_id, candidate_id),
                "claim": by_id[candidate_id].claim if candidate_id in by_id else "",
                "elo": round(tournament.ratings[candidate_id], 1),
                "shortlisted": candidate_id in tournament.shortlist_ids,
            }
            for index, candidate_id in enumerate(ordered, 1)
        ]
        packet = latest_evidence_packet(session)
        result["candidates"] = [
            _candidate_card(
                by_id[candidate_id],
                labels[candidate_id],
                reviews=reviews_by_id.get(candidate_id, []),
                rank=index,
                elo=tournament.ratings[candidate_id],
                shortlisted=candidate_id in tournament.shortlist_ids,
                citations=resolve_candidate(by_id[candidate_id], packet),
            )
            for index, candidate_id in enumerate(ordered, 1)
            if candidate_id in by_id
        ]
        rounds: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for comparison in tournament.comparisons:
            item = comparison.model_dump(mode="json")
            for key in ("candidate_a_id", "candidate_b_id", "winner_id"):
                candidate_id = item.get(key)
                item[key.replace("_id", "_label")] = labels.get(
                    candidate_id, candidate_id
                )
            rounds[comparison.round_number].append(item)
        result["comparison_rounds"] = [
            {"round": number, "comparisons": comparisons}
            for number, comparisons in sorted(rounds.items())
        ]
        result["metrics"] = [
            {"label": "Pairwise comparisons", "value": len(tournament.comparisons)},
            {"label": "Stable rounds", "value": tournament.ranking_stable_rounds},
            # Labelled "Score movement" against a bare 0.0381 this read as a rating
            # that had all but stopped moving; it is a fraction of the 1200 start,
            # which on that run was forty-six points in the final round.
            {
                "label": "Final-round rating movement",
                # The sentinel is not a fraction, and multiplied out it reported
                # "1200 points" -- a rating falling to zero in one round.
                "value": (
                    f"{round(tournament.score_movement * DEFAULT_ELO)} points"
                    if tournament.score_movement < UNMEASURED_MOVEMENT
                    else "not recorded"
                ),
            },
            {"label": "Converged", "value": tournament.converged},
        ]
        return result

    if stage == "evolve":
        payload = _latest_payload(session, "EvolutionCycle")
        if not payload:
            return None
        cycle = EvolutionCycle.model_validate(payload)
        result = _base(
            stage,
            "evolution",
            cycle.stop_reason or "Shortlisted candidates were evolved and re-reviewed",
        )
        result["evolution"] = [
            {
                "record_id": record.id,
                "parent_ids": record.parent_ids,
                "candidate": _candidate_card(
                    record.candidate,
                    f"Evolved candidate {index}",
                    citations=resolve_candidate(
                        record.candidate, latest_evidence_packet(session)
                    ),
                ),
                "changes": record.changes,
                "critiques_addressed": record.critiques_addressed,
                "new_prediction": record.new_prediction,
                "requires_rereview": record.requires_rereview,
                "round": record.round_number,
            }
            for index, record in enumerate(cycle.records, 1)
        ]
        result["metrics"] = [
            {"label": "Evolution records", "value": len(cycle.records)},
            {"label": "Independent re-reviews", "value": len(cycle.rereviews)},
            {"label": "Ranking rounds", "value": len(cycle.ranking_history)},
            {"label": "Converged", "value": cycle.converged},
        ]
        return result

    if stage == "proximity":
        payload = _latest_payload(session, "ResearchLandscape")
        if not payload:
            return None
        landscape = ResearchLandscape.model_validate(payload)
        result = _base(
            stage,
            "landscape",
            f"{len(landscape.clusters)} research clusters with explicit coverage gaps",
        )
        result["clusters"] = [
            {
                **cluster.model_dump(mode="json"),
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "label": labels.get(candidate_id, candidate_id),
                        "claim": by_id[candidate_id].claim
                        if candidate_id in by_id
                        else "",
                    }
                    for candidate_id in cluster.candidate_ids
                ],
            }
            for cluster in landscape.clusters
        ]
        result["details"] = [
            {"label": "Duplicates", "value": landscape.duplicates},
            {"label": "Coverage gaps", "value": landscape.coverage_gaps},
            {
                "label": "Protected minority hypotheses",
                "value": [
                    {
                        "label": labels.get(candidate_id, candidate_id),
                        "claim": by_id[candidate_id].claim
                        if candidate_id in by_id
                        else "",
                    }
                    for candidate_id in landscape.protected_minority_ids
                ],
            },
        ]
        return result

    if stage == "meta_review":
        payload = _latest_payload(session, "DossierManifest")
        if not payload:
            return None
        manifest = DossierManifest.model_validate(payload)
        result = _base(
            stage,
            "recommendations",
            "Final reconciliation of evidence, reviews, ranking, and fatal flaws",
        )
        fatal_ids = set(manifest.unresolved_fatal_flaw_candidate_ids)
        result["recommendations"] = [
            {
                "candidate_id": candidate_id,
                "label": labels.get(candidate_id, candidate_id),
                "claim": by_id[candidate_id].claim if candidate_id in by_id else "",
                "recommended": candidate_id in manifest.recommendation_candidate_ids,
                "excluded_for_fatal_flaw": candidate_id in fatal_ids,
                "fatal_flaws": [
                    flaw
                    for review in reviews_by_id.get(candidate_id, [])
                    for flaw in review["fatal_flaws"]
                ],
            }
            for candidate_id in dict.fromkeys(
                [
                    *manifest.recommendation_candidate_ids,
                    *manifest.unresolved_fatal_flaw_candidate_ids,
                ]
            )
        ]
        result["details"] = [
            {
                "label": "Evidence that would change the decision",
                "value": manifest.evidence_that_would_change_decision,
            },
            {
                "label": "Dossier sections",
                "value": [
                    {"key": section.key, "title": section.title}
                    for section in manifest.sections
                ],
            },
        ]
        return result
    return None
