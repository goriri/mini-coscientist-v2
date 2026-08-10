"""Resolve the evidence a candidate claims to rest on.

A generation specialist writes ``evidence_ids`` freely. Nothing downstream
checked that those ids name anything real, so a candidate could cite
``claim_001`` into an empty :class:`EvidencePacket` and every later stage --
review, ranking, dossier -- would repeat the citation as if it were grounding.
That is the one failure mode this system exists to prevent, so citations are
resolved once, here, and the unresolved ones are named rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    Candidate,
    CandidatePopulation,
    EvidenceClaim,
    EvidencePacket,
    Session,
    SourceRecord,
)

# Verification states in which a claim may be presented as evidence. Anything
# else is a lead: real text that nobody has confirmed says what we think it
# says.
GROUNDED_STATUSES = frozenset({"verified", "corrected"})

# States that actively discredit a citation rather than merely failing to
# confirm it. These outrank every other support verdict.
DISCREDITED_STATUSES = frozenset({"retracted", "inaccessible"})


@dataclass(frozen=True)
class Citation:
    """One ``evidence_ids`` entry and what, if anything, it points at."""

    reference: str
    claim: EvidenceClaim | None = None
    source: SourceRecord | None = None

    @property
    def resolved(self) -> bool:
        return self.claim is not None or self.source is not None

    @property
    def grounded(self) -> bool:
        """Resolved *and* verified. Unresolved citations are never grounded."""
        if self.claim is not None:
            return self.claim.verification_status in GROUNDED_STATUSES
        if self.source is not None:
            return self.source.verification_status in GROUNDED_STATUSES
        return False

    @property
    def discredited(self) -> bool:
        return self.resolved and self.status in DISCREDITED_STATUSES

    @property
    def status(self) -> str:
        if not self.resolved:
            return "unresolved"
        if self.claim is not None:
            return self.claim.verification_status
        assert self.source is not None
        return self.source.verification_status


@dataclass(frozen=True)
class CandidateCitations:
    candidate_id: str
    citations: list[Citation]

    @property
    def unresolved(self) -> list[str]:
        return [item.reference for item in self.citations if not item.resolved]

    @property
    def grounded(self) -> list[str]:
        return [item.reference for item in self.citations if item.grounded]

    @property
    def discredited(self) -> list[str]:
        return [item.reference for item in self.citations if item.discredited]

    @property
    def discrediting_statuses(self) -> frozenset[str]:
        """Which of the two discrediting verdicts this candidate's citations carry.

        ``support`` folds retraction and unretrievability into one word, and the
        sentence written for that word asserted retraction of both: four ideas whose
        every citation had merely failed to come back were told "those citations do
        resolve to records, and the records no longer stand". A withdrawn paper and a
        dead link are not the same fact about the literature and a reader acting on
        the warning needs to know which of them they are facing.
        """
        return frozenset(
            item.status for item in self.citations if item.discredited
        ) & frozenset(DISCREDITED_STATUSES)

    @property
    def support(self) -> str:
        """How much of this candidate's stated grounding actually exists.

        Ordered worst-first, because a reader needs the reason to distrust a
        hypothesis before the reasons to trust it. ``uncited`` is deliberately
        distinct from ``unsupported``: a candidate that claims no evidence is
        honest, whereas one that cites evidence it does not have is not.
        """
        if not self.citations:
            return "uncited"
        if self.unresolved:
            return "unsupported"
        if self.discredited:
            return "discredited"
        if self.grounded:
            return (
                "grounded"
                if len(self.grounded) == len(self.citations)
                else "partially_grounded"
            )
        return "unverified"

    @property
    def qualified(self) -> bool:
        """Whether this candidate's grounding is one the Evidence integrity list reports.

        The list prints a line per case rather than a line per idea, so nothing in it
        counts the ideas covered, and on both live runs it covered every one of them
        under a lead-in that said "the following ideas".
        """
        return self.support not in {"grounded", "partially_grounded"}


def latest_evidence_packet(session: Session) -> EvidencePacket | None:
    for artifact in reversed(session.artifacts):
        if artifact.schema_name == "EvidencePacket" and artifact.payload:
            return EvidencePacket.model_validate(artifact.payload)
    return None


def citable_ids(session: Session) -> list[str]:
    """Every evidence id a specialist is allowed to cite in this session."""
    packet = latest_evidence_packet(session)
    if packet is None:
        return []
    return [claim.id for claim in packet.claims] + [
        source.id for source in packet.sources
    ]


def citation_rule(session: Session) -> str:
    """The prompt clause that stops a specialist inventing its grounding.

    A live run showed all eight candidates citing ``claim_001`` into an empty
    evidence packet -- the packet was in the prompt, and the model cited past
    it anyway. Naming the permitted ids, or stating plainly that there are
    none, is far harder to write past than an absent list.
    """
    available = citable_ids(session)
    if not available:
        return (
            "Citable evidence: NONE. This session has no verified evidence "
            "records, so evidence_ids must be an empty list on every item. Do "
            "not invent identifiers such as claim_001; an invented citation is "
            "treated as a fabricated result."
        )
    return (
        "Citable evidence: cite only these identifiers, exactly as written, "
        "and leave evidence_ids empty where none of them applies -- "
        f"{', '.join(available)}."
    )


def resolve_candidate(
    candidate: Candidate, packet: EvidencePacket | None
) -> CandidateCitations:
    claims = {claim.id: claim for claim in packet.claims} if packet else {}
    sources = {source.id: source for source in packet.sources} if packet else {}
    return CandidateCitations(
        candidate_id=candidate.id,
        citations=[
            Citation(
                reference=reference,
                claim=claims.get(reference),
                source=sources.get(reference),
            )
            for reference in candidate.evidence_ids
        ],
    )


def resolve_population(
    population: CandidatePopulation, packet: EvidencePacket | None
) -> dict[str, CandidateCitations]:
    return {
        candidate.id: resolve_candidate(candidate, packet)
        for candidate in population.candidates
    }


def integrity_warnings(
    resolved: dict[str, CandidateCitations],
) -> list[str]:
    """Human-readable warnings for a report's provenance section.

    One line per affected candidate rather than one per citation: a reader
    needs to know which hypotheses to distrust, not to count broken ids.
    """
    warnings: list[str] = []
    for candidate_id, citations in sorted(resolved.items()):
        if citations.unresolved:
            warnings.append(
                f"{candidate_id} cites evidence that does not exist in this "
                f"session: {', '.join(citations.unresolved)}. Treat its "
                f"claim as unsupported."
            )
        elif citations.discredited:
            warnings.append(
                f"{candidate_id} cites evidence that was retracted or could "
                f"not be retrieved: {', '.join(citations.discredited)}. Treat "
                f"its claim as discredited."
            )
        elif citations.support == "unverified":
            warnings.append(
                f"{candidate_id} rests on evidence that was discovered but "
                f"never verified. Treat its claim as a hypothesis."
            )
        elif citations.support == "uncited":
            # An uncited candidate used to produce no warning at all, on the reasoning
            # that citing nothing is honest where citing what you do not have is not.
            # Honest is not the same as grounded: a live run ranked an uncited idea
            # first, and it was the one idea absent from the list of ideas whose
            # grounding does not hold.
            warnings.append(
                f"{candidate_id} cites no evidence at all, so nothing in this "
                f"session grounds it either way. Treat its claim as a conjecture."
            )
    return warnings
