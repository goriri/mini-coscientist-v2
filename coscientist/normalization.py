"""Bounded schema repair, normalization, and semantic validation for specialist outputs."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import (
    CandidatePopulation,
    DossierManifest,
    ResearchPlan,
    ReviewSet,
    Session,
    TournamentState,
)

T = TypeVar("T", bound=BaseModel)

_TEMPLATE_PLACEHOLDERS = (
    "TODO",
    "TBD",
    "[insert",
    "placeholder",
    "replace this",
    "xxx",
)


class NormalizationError(ValueError):
    """Raised when specialist output cannot be normalized or fails semantic checks."""


_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


def strip_unstorable_characters(text: str) -> str:
    """Remove the characters PostgreSQL refuses to store in JSON.

    A Deep Research report came back carrying a NUL. Every specialist turn is
    persisted by ADK as a JSON event, and PostgreSQL rejects ``\\u0000`` inside
    JSON outright -- ``asyncpg.exceptions.UntranslatableCharacterError:
    unsupported Unicode escape sequence``. The commit failed, the discovery
    specialist's call failed with it, and the evidence stage reported nothing
    discovered after seven completed passes and twenty-one dollars of research.

    Lone surrogates go the same way: they survive in a Python string, and are
    then unencodable as UTF-8 the moment anything tries to write them.

    Dropping them is safe in a way that failing is not. Neither character
    carries meaning in a research report, and the alternative on this path is
    losing the whole wave.
    """
    return _UNSTORABLE.sub("", text)


def strip_unstorable_values(value: Any) -> Any:
    """Apply :func:`strip_unstorable_characters` to every string in a structure.

    For what a tool hands back to a specialist. ``fetch_source_document``
    returns the retrieved document's own text, and a PDF is a container for
    arbitrary bytes: ADK persists the tool response as part of the event, so one
    NUL in one fetched paper failed the commit and took the source-verification
    turn down with it.
    """
    if isinstance(value, str):
        return strip_unstorable_characters(value)
    if isinstance(value, dict):
        return {key: strip_unstorable_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_unstorable_values(item) for item in value]
    return value


def repair_json_string(content: str) -> str:
    """Attempt a bounded, one-pass deterministic AST/regex repair of malformed JSON."""
    cleaned = content.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Find the outermost JSON object bounds if there is leading/trailing text
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    # Fix trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

    # Count braces and brackets and attempt to close unclosed ones
    open_braces = cleaned.count("{") - cleaned.count("}")
    open_brackets = cleaned.count("[") - cleaned.count("]")
    if open_brackets > 0:
        cleaned += "]" * open_brackets
    if open_braces > 0:
        cleaned += "}" * open_braces

    return cleaned


def try_parse_contract(content: str, model: type[T]) -> T | None:
    """Attempt to parse and validate JSON against a Pydantic contract with 1 bounded repair pass."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
            return model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            continue

    # Attempt bounded one-pass repair
    repaired = repair_json_string(content)
    try:
        return model.model_validate_json(repaired)
    except (json.JSONDecodeError, ValidationError):
        return None


def _check_string_for_placeholders(text: str) -> None:
    for placeholder in _TEMPLATE_PLACEHOLDERS:
        if placeholder in text:
            raise NormalizationError(
                f"Template leakage detected: placeholder '{placeholder}' found in output."
            )


def validate_no_template_leakage(payload: Any) -> None:
    """Scan a payload recursively for template leakage and placeholder strings."""
    if isinstance(payload, str):
        _check_string_for_placeholders(payload)
    elif isinstance(payload, dict):
        for value in payload.values():
            validate_no_template_leakage(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            validate_no_template_leakage(item)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def validate_candidate_distinctness(population: CandidatePopulation) -> None:
    """Ensure candidate population has mechanism and prediction diversity."""
    candidates = population.candidates
    for i in range(len(candidates)):
        tokens_i = _tokenize(
            candidates[i].claim
            + " "
            + candidates[i].rationale
            + " "
            + candidates[i].falsifier
        )
        for j in range(i + 1, len(candidates)):
            tokens_j = _tokenize(
                candidates[j].claim
                + " "
                + candidates[j].rationale
                + " "
                + candidates[j].falsifier
            )
            if not tokens_i or not tokens_j:
                continue
            intersection = len(tokens_i.intersection(tokens_j))
            union = len(tokens_i.union(tokens_j))
            if union > 0 and (intersection / union) > 0.85:
                raise NormalizationError(
                    f"Candidate population fails distinctness diversity check between "
                    f"{candidates[i].id} and {candidates[j].id}."
                )


def validate_candidate_comprehensiveness(population: CandidatePopulation) -> None:
    """Ensure every candidate card has substantial mechanism depth and validation design."""
    for c in population.candidates:
        if not c.title or not c.title.strip():
            raise NormalizationError(
                f"Candidate {c.id} is missing a reader-facing scientific title."
            )
        mech_words = len(re.findall(r"\w+", c.mechanism_model))
        if mech_words < 20:
            raise NormalizationError(
                f"Candidate {c.id} mechanism_model has only {mech_words} words (requires >= 20 words of substantive domain explanation)."
            )
        val_words = len(re.findall(r"\w+", c.validation_protocol))
        if val_words < 20:
            raise NormalizationError(
                f"Candidate {c.id} validation_protocol has only {val_words} words (requires >= 20 words of experimental/analytical study design)."
            )
        has_ev = bool(
            c.evidence_for or c.evidence_against or c.evidence_gaps or c.evidence_ids
        )
        if not has_ev:
            raise NormalizationError(
                f"Candidate {c.id} is missing evidence grounding (at least one of evidence_for, evidence_against, or evidence_gaps required)."
            )


def is_deterministic_fixture_content(role: str, content: str) -> bool:
    """Check if the content is from the offline DeterministicProvider CI fixture."""
    markers = {
        "goal_manager": "Research objective:",
        "evidence_discovery": "Evidence discovery status:",
        "source_verification": "Source verification status:",
        "generation": "Eight candidate records were created",
        "generation_evidence_first": "Eight candidate records were created",
        "generation_mechanism_first": "Eight candidate records were created",
        "generation_analogy_transfer": "Eight candidate records were created",
        "generation_competing_explanation": "Eight candidate records were created",
        "reflection": "Critical review:",
        "novelty_review": "Novelty review:",
        "methods_statistics": "Methods and statistics review:",
        "impact_review": "Impact review:",
        "ethics_safety_governance": "Governance review:",
        "ranking": "Prioritization (provisional):",
        "evolution": "Refined lead hypothesis:",
        "proximity": "Related-work map to validate:",
        "meta_reviewer": "Meta-review:",
    }
    marker = markers.get(role, "")
    return bool(marker and content.strip().startswith(marker))


def normalize_specialist_output(
    session: Session, role: str, content: str
) -> tuple[str, dict[str, Any]]:
    """Parse specialist output into a typed payload dict with schema repair and semantic validators."""
    if session.workflow_version >= 2 and not is_deterministic_fixture_content(
        role, content
    ):
        if role == "goal_manager":
            parsed_plan = try_parse_contract(content, ResearchPlan)
            if parsed_plan is not None:
                payload_dict = parsed_plan.model_dump(mode="json")
                validate_no_template_leakage(payload_dict)
                return "ResearchPlan", payload_dict
            raise NormalizationError(
                "Failed to normalize goal_manager response into ResearchPlan."
            )
        elif role == "ranking":
            parsed_tournament = try_parse_contract(content, TournamentState)
            if parsed_tournament is not None:
                payload_dict = parsed_tournament.model_dump(mode="json")
                validate_no_template_leakage(payload_dict)
                try:
                    pop_artifact = next(
                        item
                        for item in reversed(session.artifacts)
                        if item.schema_name == "CandidatePopulation" and item.payload
                    )
                    pop_ids = {
                        c.id
                        for c in CandidatePopulation.model_validate(
                            pop_artifact.payload
                        ).candidates
                    }
                    rank_ids = {r.candidate_id for r in parsed_tournament.rankings}
                    if rank_ids == pop_ids and len(parsed_tournament.shortlist_ids) in {
                        3,
                        4,
                        5,
                    }:
                        return "TournamentState", payload_dict
                    raise NormalizationError(
                        "Ranking tournament candidate IDs do not match candidate population exactly."
                    )
                except ValueError:
                    return "TournamentState", payload_dict
            raise NormalizationError(
                "Failed to normalize ranking response into TournamentState."
            )
        elif role in {
            "generation",
            "generation_evidence_first",
            "generation_mechanism_first",
            "generation_analogy_transfer",
            "generation_competing_explanation",
        }:
            parsed_pop = try_parse_contract(content, CandidatePopulation)
            if parsed_pop is not None:
                payload_dict = parsed_pop.model_dump(mode="json")
                validate_no_template_leakage(payload_dict)
                validate_candidate_distinctness(parsed_pop)
                validate_candidate_comprehensiveness(parsed_pop)
                return "CandidatePopulation", payload_dict
            raise NormalizationError(
                "Failed to normalize generation response into CandidatePopulation."
            )
        elif role in {
            "reflection",
            "novelty_review",
            "methods_statistics",
            "impact_review",
            "ethics_safety_governance",
        }:
            parsed_revs = try_parse_contract(content, ReviewSet)
            if parsed_revs is not None:
                payload_dict = parsed_revs.model_dump(mode="json")
                validate_no_template_leakage(payload_dict)
                return "ReviewSet", payload_dict
            raise NormalizationError(
                f"Failed to normalize {role} response into ReviewSet."
            )
        elif role == "meta_reviewer":
            parsed_manifest = try_parse_contract(content, DossierManifest)
            if parsed_manifest is not None:
                payload_dict = parsed_manifest.model_dump(mode="json")
                validate_no_template_leakage(payload_dict)
                return "DossierManifest", payload_dict
            from .parity import dossier_manifest

            return "DossierManifest", dossier_manifest(session).model_dump(mode="json")
        elif role == "evidence_discovery":
            from .parity import evidence_packet

            return "EvidencePacket", evidence_packet(
                session, content, verified=False
            ).model_dump(mode="json")
        elif role == "source_verification":
            from .parity import evidence_packet

            return "EvidencePacket", evidence_packet(
                session, content, verified=True
            ).model_dump(mode="json")

    # For offline CI fixtures and V1 regression tests, delegate to deterministic parity functions
    from .parity import (
        candidate_population,
        dossier_manifest,
        evidence_packet,
        evolution_cycle,
        parsed_review_set,
        research_landscape,
        research_plan,
        tournament_state,
    )

    if role == "goal_manager":
        value: BaseModel = research_plan(session)
    elif role == "evidence_discovery":
        value = evidence_packet(session, content, verified=False)
    elif role == "source_verification":
        value = evidence_packet(session, content, verified=True)
    elif role in {
        "generation",
        "generation_evidence_first",
        "generation_mechanism_first",
        "generation_analogy_transfer",
        "generation_competing_explanation",
    }:
        value = candidate_population(session, content)
    elif role in {
        "reflection",
        "novelty_review",
        "methods_statistics",
        "impact_review",
        "ethics_safety_governance",
    }:
        value = parsed_review_set(session, role, content)
    elif role == "ranking":
        value = tournament_state(session)
    elif role == "evolution":
        value = evolution_cycle(session)
    elif role == "proximity":
        value = research_landscape(session)
    elif role == "meta_reviewer":
        value = dossier_manifest(session)
    else:
        raise ValueError(f"No typed specialist contract for role: {role}")

    return type(value).__name__, value.model_dump(mode="json")
