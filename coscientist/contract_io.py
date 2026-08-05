"""Schema-derived prompts and tolerant parsing for typed specialist output.

The specialist contracts in :mod:`coscientist.models` are strict
(``extra="forbid"``) because they are service boundaries. A language model,
however, reliably invents near-miss field names -- ``prediction`` for
``predictions``, ``category`` for ``generation_strategy``, ``score`` for
``confidence``. Under a strict contract a single near-miss discards the whole
payload and the caller silently substitutes a generic template, so a stage that
did real scientific work renders as boilerplate.

This module removes both halves of that failure:

* :func:`contract_skeleton` renders the contract itself into the prompt, so an
  instruction built from it cannot drift away from the schema it describes.
* :func:`parse_contract` repairs truncation, normalizes known aliases, drops
  unknown keys, and coerces obvious type mismatches before validating. The
  contract stays strict; only the LLM-facing edge is forgiving.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Field renames a specialist may use. Keys are lowercase source names; values
# are the contract field they map onto. A rename is applied only when the target
# field exists on the model being parsed and the source name is not itself a
# field of that model, so the same table can serve every contract.
_ALIASES: dict[str, str] = {
    # Candidate
    "category": "generation_strategy",
    "strategy": "generation_strategy",
    "generation_type": "generation_strategy",
    "prediction": "predictions",
    "testable_predictions": "predictions",
    "alternative": "alternatives",
    "alternative_explanations": "alternatives",
    "competing_explanations": "alternatives",
    "falsifiers": "falsifier",
    "falsification_criterion": "falsifier",
    "hypothesis": "claim",
    "reasoning": "rationale",
    "justification": "rationale",
    "risk": "risks",
    "dependency": "dependencies",
    "go_no_go": "go_no_go_tests",
    "go_no_go_test": "go_no_go_tests",
    "evidence_id": "evidence_ids",
    # CandidateReview
    "review_type": "criterion",
    "criteria": "criterion",
    "dimension": "criterion",
    "critique": "findings",
    "finding": "findings",
    "strengths": "findings",
    "prior_art_summary": "findings",
    "novelty_assessment": "findings",
    "information_gain": "findings",
    "importance": "findings",
    "feasibility": "findings",
    "weaknesses": "objections",
    "concerns": "objections",
    "incrementalism_assessment": "objections",
    "external_validity": "objections",
    "recommendations": "recommendation",
    "verdict": "recommendation",
    "decision": "recommendation",
    "score": "confidence",
    "reviewer_role": "reviewer",
    "agent": "reviewer",
    "fatal_flaw": "fatal_flaws",
    "blocking_flaws": "fatal_flaws",
    "critical_flaws": "fatal_flaws",
    "rebuttal": "rebuttals",
    "assumption": "assumptions",
    "unverified_claims": "assumptions",
    # PairwiseComparison
    "candidate_a": "candidate_a_id",
    "candidate_b": "candidate_b_id",
    "winner": "winner_id",
    "round": "round_number",
    "presented_first": "presented_first_id",
    "scores": "criterion_scores",
    "reason": "rationale",
    # TournamentState
    "shortlist": "shortlist_ids",
    "elo_ratings": "ratings",
    "final_ratings": "ratings",
    # ResearchCluster
    "cluster_name": "name",
    "label": "name",
    "mechanism": "shared_mechanism",
    "outcome": "shared_outcome",
    "candidates": "candidate_ids",
    "members": "candidate_ids",
    "data_needs": "required_data",
    "required_datasets": "required_data",
    # DossierManifest
    "recommended_candidates": "recommendation_candidate_ids",
    "excluded_candidates": "unresolved_fatal_flaw_candidate_ids",
    "evidence_requirements": "evidence_that_would_change_decision",
    # EvidencePacket
    "verified_claims": "claims",
    "source": "sources",
    "claim_records": "claims",
    "limitation": "limitations",
    # EvolutionRecord
    "change": "changes",
    "critique_addressed": "critiques_addressed",
    "parent_id": "parent_ids",
    "evolved_candidate": "candidate",
    # No contract field holds a procedure; the go/no-go tests are the closest slot.
    "protocol": "go_no_go_tests",
}

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _ratings_from_candidate_map(payload: dict[str, Any]) -> dict[str, Any]:
    """Lift ``{"candidates": {id: {"elo": n}}}`` into ``TournamentState.ratings``.

    Specialists frequently report the tournament as a per-candidate record
    rather than the flat rating map the contract declares. The numbers are
    right there; only the shape differs.
    """
    holder = payload.get("candidates")
    if "ratings" in payload or not isinstance(holder, dict):
        return payload
    ratings: dict[str, float] = {}
    for candidate_id, record in holder.items():
        if not isinstance(record, dict):
            continue
        for key in ("elo", "elo_rating", "rating", "score"):
            if isinstance(record.get(key), (int, float)):
                ratings[candidate_id] = float(record[key])
                break
    if not ratings:
        return payload
    lifted = dict(payload)
    lifted.pop("candidates", None)
    lifted["ratings"] = ratings
    return lifted


def _records_from_rounds(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``{"rounds": [{"candidates": [...], "reviews": [...]}]}``.

    ``EvolutionCycle`` stores a flat ``records``/``rereviews`` pair with the
    round on each entry, but specialists naturally group their output by round
    because the task is described in rounds. The grouping carries the same
    information, so unwrap it rather than discard the cycle.
    """
    rounds = payload.get("rounds")
    if "records" in payload or not isinstance(rounds, list):
        return payload
    records: list[dict[str, Any]] = []
    rereviews: list[Any] = []
    for position, entry in enumerate(rounds, start=1):
        if not isinstance(entry, dict):
            continue
        number = entry.get("round_number", position)
        for candidate in entry.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            parents = candidate.get("parent_ids") or candidate.get("parent_id") or []
            records.append(
                {
                    "parent_ids": [parents] if isinstance(parents, str) else parents,
                    "candidate": candidate,
                    "changes": candidate.get("changes") or [],
                    "new_prediction": candidate.get("new_prediction")
                    or candidate.get("falsifier")
                    or "",
                    "round_number": number,
                }
            )
        rereviews.extend(entry.get("reviews") or entry.get("rereviews") or [])
    if not records:
        return payload
    flattened = dict(payload)
    flattened.pop("rounds", None)
    flattened["records"] = records
    if rereviews:
        flattened["rereviews"] = rereviews
    return flattened


# Structural fixes applied before field renaming, keyed by contract name. These
# handle reshapes that a field alias cannot express.
_PRENORMALIZE: dict[str, Any] = {
    "TournamentState": _ratings_from_candidate_map,
    "EvolutionCycle": _records_from_rounds,
}


_CRITERION_BY_REVIEWER = {
    "reflection": "evidence_correctness",
    "evidence": "evidence_correctness",
    "correctness": "evidence_correctness",
    "novelty": "novelty",
    "originality": "novelty",
    "methods": "methods_feasibility",
    "statistics": "methods_feasibility",
    "feasibility": "methods_feasibility",
    "impact": "impact_safety",
    "safety": "safety_governance",
    "ethics": "safety_governance",
    "governance": "safety_governance",
}


def _criterion_from_review(payload: dict[str, Any]) -> str:
    """Infer which axis a review judged from the reviewer that signed it.

    Reviews nested inside a larger contract -- an evolution cycle's re-reviews,
    for instance -- routinely carry the reviewer's name but not the criterion,
    which is redundant with it. Losing the whole cycle over that is absurd.
    """
    reviewer = str(payload.get("reviewer") or payload.get("criterion") or "").lower()
    for token, criterion in _CRITERION_BY_REVIEWER.items():
        if token in reviewer:
            return criterion
    return "evidence_correctness"


def _recommendation_from_review(payload: dict[str, Any]) -> str:
    """Derive a review verdict when the reviewer scored but never labelled one."""
    if payload.get("fatal_flaws"):
        return "reject"
    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.5:
        return "insufficient_evidence"
    return "revise"


# Defaults for fields a specialist routinely omits because they are context or
# bookkeeping rather than judgement. Values may be callables receiving the
# already-renamed payload. Every application is recorded as a repair.
_FIELD_DEFAULTS: dict[str, dict[str, Any]] = {
    "CandidateReview": {
        "recommendation": _recommendation_from_review,
        "criterion": _criterion_from_review,
        "reviewer": lambda payload: str(payload.get("criterion") or "reviewer"),
    },
    "ResearchCluster": {
        "name": lambda payload: (
            str(payload.get("shared_mechanism", "Cluster")).split(".")[0].strip()[:80]
            or "Cluster"
        ),
        "candidate_ids": [],
        "shared_mechanism": "",
        "shared_outcome": "",
    },
    "PairwiseComparison": {
        "presented_first_id": lambda payload: payload.get("candidate_a_id", ""),
        "rationale": "",
        "round_number": 1,
    },
    "EvolutionRecord": {"parent_ids": [], "changes": [], "new_prediction": ""},
    "Candidate": {"rationale": "", "falsifier": ""},
    "SourceRecord": {"url": ""},
    "EvidenceClaim": {"claim": ""},
}


@dataclass
class ParseOutcome:
    """Result of parsing one specialist response into its contract."""

    value: BaseModel | None = None
    repairs: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


# --------------------------------------------------------------------------- #
# Prompt generation
# --------------------------------------------------------------------------- #


def _type_label(annotation: Any, nested: dict[str, type[BaseModel]]) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        return "|".join(json.dumps(arg) for arg in get_args(annotation))
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        inner = "|".join(_type_label(arg, nested) for arg in args)
        return f"{inner}|null" if len(args) < len(get_args(annotation)) else inner
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        return f"[{_type_label(args[0], nested) if args else 'any'}]"
    if origin is dict:
        args = get_args(annotation)
        key = _type_label(args[0], nested) if args else "str"
        value = _type_label(args[1], nested) if len(args) > 1 else "any"
        return f"{{{key}: {value}}}"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        nested[annotation.__name__] = annotation
        return annotation.__name__
    return getattr(annotation, "__name__", str(annotation))


def contract_skeleton(model: type[BaseModel]) -> str:
    """Render ``model`` and its nested contracts as a compact prompt schema.

    The rendering is derived from the contract at call time, so an instruction
    built from it cannot describe a field the contract does not have.
    """
    rendered: dict[str, str] = {}
    queue: list[type[BaseModel]] = [model]
    seen: set[str] = set()
    while queue:
        current = queue.pop(0)
        if current.__name__ in seen:
            continue
        seen.add(current.__name__)
        nested: dict[str, type[BaseModel]] = {}
        lines = [f"{current.__name__} {{"]
        for name, info in current.model_fields.items():
            if name == "schema_version":
                continue
            label = _type_label(info.annotation, nested)
            if info.is_required():
                suffix = "   // REQUIRED"
            elif info.default_factory is None and info.default is not None:
                suffix = f" = {json.dumps(info.default, default=str)}"
            else:
                suffix = ""
            lines.append(f"  {name}: {label}{suffix}")
        lines.append("}")
        rendered[current.__name__] = "\n".join(lines)
        queue.extend(nested.values())
    ordered = [rendered.pop(model.__name__)]
    ordered.extend(rendered[name] for name in sorted(rendered))
    return "\n\n".join(ordered)


def schema_instruction(model: type[BaseModel], extra: str = "") -> str:
    """Build the structured-output instruction for one specialist role."""
    body = (
        f"Return exactly one JSON object matching the {model.__name__} contract "
        "below. Emit raw JSON only: no prose before or after it, no markdown "
        "code fence, no comments.\n\n"
        f"{contract_skeleton(model)}\n\n"
        "Rules: use these field names exactly and add no others; a field typed "
        "[x] is always a JSON array, never a bare string; a field whose type is "
        'written as "a"|"b" must contain one of those literal strings verbatim; '
        "every field marked REQUIRED must be present and non-empty."
    )
    return f"{body}\n\n{extra.strip()}" if extra.strip() else body


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _close_truncated(fragment: str) -> str | None:
    """Close a JSON object that an output-token limit cut off mid-value.

    Returns the longest prefix that ends on a complete value, re-closed with the
    containers that were open *at that point*. Closing with the final stack
    instead would mismatch the truncation point and produce invalid JSON.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    checkpoint: tuple[int, tuple[str, ...]] | None = None
    for index, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
            if not stack:
                return None  # balanced already; truncation is not the problem
            checkpoint = (index + 1, tuple(stack))
        elif char == ",":
            checkpoint = (index, tuple(stack))
    if not stack or checkpoint is None:
        return None
    position, open_containers = checkpoint
    return fragment[:position].rstrip() + "".join(reversed(open_containers))


_MAX_FRAGMENTS = 100


def _candidate_fragments(content: str) -> list[str]:
    """Yield plausible JSON fragments from a raw specialist response.

    Fenced blocks come first, then every ``{`` in document order. The outermost
    object is therefore tried early, but inner objects stay available as
    fallbacks for a response whose wrapper is malformed.
    """
    # Both closed fences and a fence the model never got to close, because a
    # response cut off by the output-token limit ends mid-object inside it.
    fragments = [
        block.strip()
        for block in re.findall(r"```(?:json)?\s*(.+?)(?:```|\Z)", content, re.S)
    ]
    fragments.extend(
        content[index:] for index, char in enumerate(content) if char == "{"
    )
    return fragments[:_MAX_FRAGMENTS]


def _decode(fragment: str) -> tuple[Any, int] | None:
    """Decode a leading JSON value and report how much input it consumed."""
    decoder = json.JSONDecoder()
    for text in (fragment, _TRAILING_COMMA_RE.sub(r"\1", fragment)):
        try:
            payload, end = decoder.raw_decode(text)
            return payload, end
        except json.JSONDecodeError:
            continue
    return None


def _informativeness(value: BaseModel) -> int:
    """Count the populated, non-default leaves of a validated contract.

    Several contracts (``ReviewSet``, ``TournamentState``, ``ResearchLandscape``)
    have a default for every field, so an unrelated inner object -- or ``{}`` --
    validates against them cleanly while carrying no content. Ranking parses by
    informativeness keeps such a degenerate match from beating the real payload.
    """

    def count(node: Any) -> int:
        if isinstance(node, dict):
            return sum(count(item) for item in node.values())
        if isinstance(node, list):
            return sum(count(item) for item in node)
        return 1 if node not in (None, "", [], {}) else 0

    return count(value.model_dump(mode="json", exclude_defaults=True))


def _literal_options(annotation: Any) -> list[Any]:
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    if get_origin(annotation) in (Union, UnionType):
        for arg in get_args(annotation):
            options = _literal_options(arg)
            if options:
                return options
    return []


def _inner_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _inner_model(arg)
        if found is not None:
            return found
    return None


def _wants_list(annotation: Any) -> bool:
    if get_origin(annotation) in (list, tuple, set, frozenset):
        return True
    if get_origin(annotation) in (Union, UnionType):
        return any(_wants_list(arg) for arg in get_args(annotation))
    return False


def _wants_scalar_str(annotation: Any) -> bool:
    if annotation is str:
        return True
    if get_origin(annotation) in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return bool(args) and all(arg is str for arg in args)
    return False


# Vocabulary a specialist uses for an enum when it does not echo the contract's
# own wording. Mapped onto the contract term only when that term is allowed.
_LITERAL_SYNONYMS: dict[str, str] = {
    "approve": "advance",
    "approved": "advance",
    "accept": "advance",
    "go": "advance",
    "pass": "advance",
    "promote": "advance",
    "proceed": "advance",
    "no_go": "reject",
    "block": "reject",
    "fail": "reject",
    "decline": "reject",
    "conditional": "revise",
    "conditional_go": "revise",
    "modify": "revise",
    "improve": "revise",
    "rework": "revise",
    "insufficient": "insufficient_evidence",
    "unverified": "insufficient_evidence",
    "unknown": "insufficient_evidence",
    "needs_evidence": "insufficient_evidence",
    "inconclusive": "insufficient_evidence",
}


def _match_literal(value: Any, options: list[Any], repairs: list[str]) -> Any:
    """Snap a near-miss enum string onto the closest allowed literal."""
    if value in options or not isinstance(value, str):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    strings = [option for option in options if isinstance(option, str)]
    synonym = _LITERAL_SYNONYMS.get(normalized)
    if synonym in strings:
        repairs.append(f"literal {value!r} -> {synonym!r}")
        return synonym
    for option in strings:
        if normalized == option:
            repairs.append(f"literal {value!r} -> {option!r}")
            return option
    for option in strings:
        if normalized in option or option in normalized:
            repairs.append(f"literal {value!r} -> {option!r}")
            return option
    tokens = set(normalized.split("_"))
    ranked = sorted(strings, key=lambda o: -len(tokens & set(o.split("_"))))
    if ranked and tokens & set(ranked[0].split("_")):
        repairs.append(f"literal {value!r} -> {ranked[0]!r}")
        return ranked[0]
    return value


def _flatten_text(value: Any) -> list[str]:
    """Turn a nested LLM prose structure into a flat list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        return [item for entry in value for item in _flatten_text(entry)]
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, entry in value.items()
            for item in _flatten_text(entry)
        ]
    return [str(value)]


def _coerce_scalar(value: Any, annotation: Any, name: str, repairs: list[str]) -> Any:
    options = _literal_options(annotation)
    if options:
        if isinstance(value, str):
            return _match_literal(value, options, repairs)
        # A reviewer answering an enum with a list ("recommendations": [...]).
        # Take the first entry that names an allowed value; otherwise clear the
        # field so a derived default can stand in.
        for entry in _flatten_text(value):
            matched = _match_literal(entry, options, [])
            if matched in options:
                repairs.append(f"{name}: selected {matched!r} from a list")
                return matched
        repairs.append(f"{name}: no allowed value in {type(value).__name__}")
        return None
    if _wants_scalar_str(annotation) and not isinstance(value, str):
        repairs.append(f"{name}: collapsed {type(value).__name__} to a string")
        return "; ".join(_flatten_text(value))
    if name == "confidence" and isinstance(value, (int, float)) and value > 1:
        # Reviewers routinely answer on a 1-5 or 0-100 scale.
        scaled = max(0.0, min(1.0, value / 100 if value > 5 else value / 5))
        repairs.append(f"confidence {value} -> {round(scaled, 3)}")
        return scaled
    return value


def _coerce(
    payload: Any,
    model: type[BaseModel],
    repairs: list[str],
    defaults: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Normalize one LLM-produced object against ``model``'s field structure."""
    if not isinstance(payload, dict):
        return payload
    prenormalize = _PRENORMALIZE.get(model.__name__)
    if prenormalize is not None:
        reshaped = prenormalize(payload)
        if reshaped is not payload:
            repairs.append(f"{model.__name__}: reshaped to the declared structure")
            payload = reshaped
    fields = model.model_fields
    renamed: dict[str, Any] = {}
    merges: dict[str, list[Any]] = {}
    for key, value in payload.items():
        if key in fields:
            renamed[key] = value
            continue
        target = _ALIASES.get(key.lower())
        if target and target in fields:
            if target in renamed:
                merges.setdefault(target, []).append(value)
            else:
                renamed[target] = value
                repairs.append(f"{model.__name__}.{key} -> {target}")
            continue
        if key != "schema_version":
            repairs.append(f"{model.__name__}: dropped unknown field {key!r}")
    for target, extras in merges.items():
        if _wants_list(fields[target].annotation):
            renamed[target] = _flatten_text(renamed[target]) + _flatten_text(extras)
            repairs.append(f"{model.__name__}.{target}: merged aliased values")

    coerced: dict[str, Any] = {}
    for name, value in renamed.items():
        annotation = fields[name].annotation
        nested = _inner_model(annotation)
        if _wants_list(annotation):
            if value is None:
                continue
            items = value if isinstance(value, list) else [value]
            if not isinstance(value, list):
                repairs.append(f"{model.__name__}.{name}: wrapped a scalar in a list")
            if nested is not None:
                coerced[name] = [
                    _coerce(item, nested, repairs, defaults) for item in items
                ]
            else:
                args = get_args(annotation)
                item_type = args[0] if args else str
                coerced[name] = (
                    _flatten_text(items) if _wants_scalar_str(item_type) else items
                )
            continue
        if nested is not None and isinstance(value, dict):
            coerced[name] = _coerce(value, nested, repairs, defaults)
            continue
        if get_origin(annotation) is dict and not isinstance(value, dict):
            repairs.append(f"{model.__name__}.{name}: dropped a non-mapping value")
            continue
        coerced[name] = _coerce_scalar(value, annotation, name, repairs)
    return _fill_defaults(coerced, model, defaults, repairs)


def _fill_defaults(
    payload: dict[str, Any],
    model: type[BaseModel],
    defaults: dict[str, dict[str, Any]] | None,
    repairs: list[str],
) -> dict[str, Any]:
    """Supply required fields the specialist omitted, recording each fill."""
    supplied = dict(_FIELD_DEFAULTS.get(model.__name__, {}))
    supplied.update((defaults or {}).get(model.__name__, {}))
    for name, default in supplied.items():
        info = model.model_fields.get(name)
        if info is None or payload.get(name) not in (None, "", [], {}):
            continue
        if not info.is_required() and name not in (defaults or {}).get(
            model.__name__, {}
        ):
            continue  # optional and uncustomized: let the contract's default stand
        payload[name] = default(payload) if callable(default) else default
        repairs.append(f"{model.__name__}.{name}: filled from context")
    return payload


def parse_contract(
    content: str,
    model: type[T],
    *,
    defaults: dict[str, dict[str, Any]] | None = None,
) -> ParseOutcome:
    """Parse a specialist response into ``model``, repairing common LLM slips.

    ``defaults`` maps a contract name to field values that fill required context
    the specialist has no reason to restate -- ``question`` on an
    ``EvidencePacket``, ``reviewer`` on a ``CandidateReview``. Values may be
    callables receiving the parsed object. They apply only where the response
    left a field empty, and every application is recorded as a repair.

    Strict validation is always the final step, so anything returned here
    already satisfies the contract. Every candidate fragment is ranked by how
    much content it actually carries, because a contract whose fields all have
    defaults would otherwise accept the first stray inner object as a valid but
    empty payload.
    """
    if not content or not content.strip():
        return ParseOutcome(error="empty specialist response")
    last_error = "no JSON object found in the specialist response"
    best: tuple[int, int, ParseOutcome] | None = None
    for fragment in _candidate_fragments(content):
        repairs: list[str] = []
        decoded = _decode(fragment)
        if decoded is None:
            closed = _close_truncated(fragment)
            decoded = _decode(closed) if closed is not None else None
            if decoded is None:
                continue
            repairs.append("closed a truncated JSON object")
        payload, consumed = decoded
        if not isinstance(payload, dict):
            continue
        attempts = (
            (payload, []),
            (_coerce(payload, model, repairs, defaults), repairs),
        )
        for candidate, applied in attempts:
            try:
                value = model.model_validate(candidate)
            except ValidationError as exc:
                last_error = summarize_errors(exc)
                continue
            score = (_informativeness(value), consumed)
            if best is None or score > best[:2]:
                best = (*score, ParseOutcome(value=value, repairs=list(applied)))
            break
    if best is not None and best[0] > 0:
        return best[2]
    return ParseOutcome(error=last_error)


def summarize_errors(exc: ValidationError, limit: int = 10) -> str:
    """Collapse a Pydantic error list into a short, de-duplicated summary."""
    seen: list[str] = []
    for error in exc.errors():
        location = ".".join(
            str(part) for part in error["loc"] if not isinstance(part, int)
        )
        entry = f"{location or '<root>'}: {error['type']}"
        if entry not in seen:
            seen.append(entry)
        if len(seen) >= limit:
            break
    return "; ".join(seen)


def repair_prompt(model: type[BaseModel], content: str, error: str) -> str:
    """Build a follow-up prompt asking a specialist to fix its own JSON."""
    excerpt = content if len(content) <= 8000 else f"{content[:8000]}\n...[truncated]"
    return (
        f"Your previous response could not be parsed as a valid {model.__name__} "
        f"JSON object.\n\nValidation errors:\n{error}\n\n"
        f"Your previous response:\n{excerpt}\n\n"
        "Return the same scientific content again, corrected to satisfy the "
        "contract exactly. Keep every substantive claim, review, rationale, and "
        "identifier from your previous answer; change only the JSON structure, "
        "field names, literal values, and any truncation. If your previous "
        "answer was cut off, shorten the prose so the whole object fits.\n\n"
        f"{schema_instruction(model)}"
    )
