"""Deterministic, human-readable views over validated research artifacts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .citations import (
    CandidateCitations,
    latest_evidence_packet,
    resolve_candidate,
)
from .models import (
    Candidate,
    CandidatePopulation,
    DiscoveryManifest,
    DossierManifest,
    EvidencePacket,
    EvolutionCycle,
    ResearchLandscape,
    ResearchPlan,
    ReviewSet,
    Session,
    TournamentState,
)
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
        result = _base(
            stage,
            "evidence",
            # Naming the wrong provider is worse than naming none: a reader who
            # sees "Deep Research" trusts the leads further than search hits earn.
            f"Knowledge landscape from {', '.join(providers)} — discovered, not "
            "yet verified"
            if providers
            else "Knowledge landscape — nothing was discovered",
        )
        result["metrics"] = [
            {"label": "Deep Research passes", "value": len(manifest.runs)},
            {"label": "Source leads", "value": len(manifest.source_leads)},
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
                        "status": run.status,
                        "elapsed": [run.started_at, run.completed_at],
                        "cost": run.estimated_cost_usd,
                        "error": run.error,
                    }
                    for run in manifest.runs
                ],
            },
            {
                "label": "Coverage by facet",
                "value": latest.facet_scores if latest else {},
            },
            {
                "label": "Unresolved gaps",
                "value": [
                    {
                        "facet": gap.facet,
                        "description": gap.description,
                        "impact": gap.decision_impact,
                    }
                    for gap in (latest.gaps if latest else [])
                ],
            },
            {
                "label": "Source leads",
                "value": [
                    {
                        "title": lead.title,
                        "url": lead.canonical_url,
                        "type": lead.source_type,
                        "passes": lead.originating_passes,
                        "status": lead.verification_status,
                    }
                    for lead in manifest.source_leads
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
