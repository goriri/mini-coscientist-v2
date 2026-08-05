"""A session rich enough to exercise the report structure end to end.

The stored fixtures on disk come from runs with Deep Research switched off, so they
carry no source leads, no debate transcripts and no failed payloads. Those are exactly
the states the report has to render carefully, so they are built here instead of being
waited for.
"""

from __future__ import annotations

import pytest

from coscientist.models import (
    Artifact,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EvidenceClaim,
    EvidencePacket,
    PairwiseComparison,
    ResearchPlan,
    ReviewSet,
    Session,
    SourceLead,
    SourceRecord,
    TournamentState,
)

_LEADS = [
    ("Thin-film passivation of silicon anodes - PubMed", "verified", 0.9),
    (
        "Binder chemistry and pore blockage in composite electrodes - PMC",
        "verified",
        0.6,
    ),
    ("Solid electrolyte interphase - Wikipedia", "discovered_unverified", 0.4),
    (
        "Cycle-life benchmarking across coating thicknesses | ResearchGate",
        "retracted",
        0.5,
    ),
    ("Fluoroethylene carbonate additives in lithium cells - PubMed", "verified", 0.85),
    ("Operando dilatometry of coated anodes - PMC", "inaccessible", 0.3),
    ("Scavenger additives and trace water in electrolytes - PubMed", "verified", 0.95),
    ("Mechanical failure modes of thick coatings - PMC", "verified", 0.55),
]


# Discovery findings and evidence claims are full sentences in a real run, and their
# length is load-bearing: the narrative's word budget is fed from them, so a fixture
# of five-word stubs would understate the report rather than exercise it.
_STATEMENTS = [
    "Atomic layer deposition of alumina on silicon anodes suppresses continuous "
    "electrolyte reduction for the first fifty cycles, after which the effect "
    "plateaus and capacity fade resumes at close to the uncoated rate",
    "Binder selection changes the accessible pore volume of a composite electrode by "
    "up to a third, which is large enough to account for reported cycle-life gains "
    "without invoking any interphase mechanism at all",
    "The solid electrolyte interphase is conventionally described as a two-layer "
    "structure whose inner inorganic layer governs lithium transport while the outer "
    "organic layer governs electrolyte access",
    "Benchmarking across published coating thicknesses shows no monotonic "
    "relationship between thickness and retention once cell format and electrolyte "
    "composition are controlled for",
    "Fluoroethylene carbonate additives improve first-cycle coulombic efficiency by "
    "several percentage points in silicon-containing cells, an effect that overlaps "
    "with and can mask the contribution of a deposited coating",
    "Operando dilatometry indicates that coated electrodes expand less on the first "
    "lithiation, though the reported traces are noisy enough that the difference is "
    "not resolved at the stated confidence",
    "Trace water in the electrolyte generates hydrofluoric acid that attacks the "
    "interphase, and scavenger additives that remove it produce retention gains "
    "comparable to those attributed to coatings",
    "Thick coatings fail mechanically rather than chemically, cracking under the "
    "volume change of the underlying particle and exposing fresh surface at a rate "
    "that increases with thickness",
]
_CLAIMS = [
    "A conformal alumina layer of five to ten nanometres reduces first-cycle "
    "irreversible capacity loss in silicon-containing anodes",
    "Pore blockage by the binder is a sufficient explanation for the retention "
    "improvements reported in coated-electrode studies",
    "Interphase composition rather than interphase thickness determines the rate of "
    "lithium transport across the passivating layer",
    "Coating thickness above twenty nanometres provides no further retention benefit "
    "in any published cell format",
    "Fluorinated carbonate additives and deposited coatings act on the same failure "
    "mode and their benefits are therefore not additive",
    "Electrode expansion on first lithiation is reduced by a deposited coating, "
    "though the published traces do not resolve the difference",
    "Removing trace water from the electrolyte reproduces most of the retention "
    "benefit attributed to surface coatings",
    "Mechanical cracking of the coating, not its chemical breakdown, sets the upper "
    "bound on useful coating thickness",
]


def _url(index: int) -> str:
    return (
        f"https://vertexaisearch.cloud.google.com/grounding-api-redirect/lead-{index}"
    )


def _candidates() -> list[Candidate]:
    specs = [
        (
            "A conformal alumina coating suppresses electrolyte decomposition at the "
            "silicon anode surface",
            "evidence_first",
            ["claim_1"],
        ),
        (
            "Test whether a fluorinated binder redistributes lithium flux across the "
            "composite electrode",
            "mechanism_first",
            ["claim_2"],
        ),
        (
            "Transfer the scavenger-additive strategy from sodium cells to the "
            "silicon anode system",
            "analogy_transfer",
            [],
        ),
        (
            "The observed capacity retention is explained by pore blockage rather "
            "than by interphase stabilisation",
            "competing_explanation",
            ["claim_missing"],
        ),
        (
            "Cycle-life gains persist once the coating thickness exceeds twenty "
            "nanometres",
            "evidence_first",
            ["claim_3"],
        ),
        (
            "Trace water scavenging and surface passivation contribute "
            "independently to first-cycle efficiency",
            "mechanism_first",
            ["claim_0", "claim_2"],
        ),
    ]
    return [
        Candidate(
            id=f"candidate_{index:04d}",
            claim=claim,
            rationale=f"Rationale {index}: the coating alters the interphase in a way "
            "that is measurable before capacity fade appears",
            predictions=[f"Prediction {index}a", f"Prediction {index}b"],
            alternatives=[f"Alternative reading {index}"],
            falsifier=f"No difference in first-cycle coulombic efficiency at {index} nm",
            dependencies=[f"Dependency {index}"],
            risks=[f"Risk {index}"],
            go_no_go_tests=[f"Go/no-go threshold {index}"],
            generation_strategy=strategy,
            evidence_ids=evidence_ids,
        )
        for index, (claim, strategy, evidence_ids) in enumerate(specs, start=1)
    ]


_REVIEW_SPECS = [
    (
        "evidence_correctness",
        "reflection",
        "advance",
        "The cited measurements were taken on comparable cell chemistries",
        "The effect size is reported without a dispersion estimate",
        "A dispersion estimate can be recovered from the deposited raw traces",
    ),
    (
        "novelty",
        "novelty_review",
        "revise",
        "Coating strategies are well represented in the prior literature",
        "The mechanism restates an established passivation account",
        "",
    ),
    (
        "methods_feasibility",
        "methods_statistics",
        "advance",
        "The decisive measurement runs on equipment the laboratory already owns",
        "The sample size will not resolve a difference below five per cent",
        "Doubling the cell count keeps the study inside a single quarter",
    ),
    (
        "impact_safety",
        "impact_review",
        "insufficient_evidence",
        "A confirmed result would change how cells are formatted at scale",
        "Nothing in the record quantifies the manufacturing cost of the change",
        "",
    ),
    (
        "safety_governance",
        "ethics_safety_governance",
        "revise",
        "No step in the protocol exceeds the rated voltage window",
        "Thermal runaway screening has not been scheduled",
        "A safety review can be attached to the existing cell-build gate",
    ),
]


def _reviews(candidates: list[Candidate]) -> ReviewSet:
    return ReviewSet(
        reviews=[
            CandidateReview(
                id=f"review_{candidate.id}_{index}",
                candidate_id=candidate.id,
                criterion=criterion,
                reviewer=reviewer,
                recommendation=recommendation,
                findings=[finding],
                objections=[objection],
                rebuttals=[rebuttal] if rebuttal else [],
                confidence=0.5 + 0.05 * index,
            )
            for candidate in candidates
            for index, (
                criterion,
                reviewer,
                recommendation,
                finding,
                objection,
                rebuttal,
            ) in enumerate(_REVIEW_SPECS)
        ]
    )


def _tournament(candidates: list[Candidate]) -> TournamentState:
    pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (4, 0)]
    comparisons = []
    for round_number, (left, right) in enumerate(pairs, start=1):
        a, b = candidates[left], candidates[right]
        debated = round_number % 2 == 1
        comparisons.append(
            PairwiseComparison(
                id=f"comparison_{round_number}",
                round_number=round_number,
                candidate_a_id=a.id,
                candidate_b_id=b.id,
                presented_first_id=a.id,
                winner_id=a.id,
                rationale=f"The first idea carried the stronger mechanism in round "
                f"{round_number}",
                confidence=0.7,
                elo_before={a.id: 1500.0, b.id: 1500.0},
                elo_after={a.id: 1516.0, b.id: 1484.0},
                debate_turns=(
                    [
                        f"Turn 1: the case for the first idea in round {round_number}",
                        f"Turn 2: the case for the second idea in round {round_number}",
                        '{"verdict": "a", "score": 1}',
                    ]
                    if debated
                    else []
                ),
                judge="llm_debate" if debated else "deterministic",
            )
        )
    return TournamentState(
        ratings={
            candidate.id: 1600.0 - 25.0 * index
            for index, candidate in enumerate(candidates)
        },
        comparisons=comparisons,
        shortlist_ids=[candidates[0].id, candidates[1].id],
        swiss_rounds=2,
        top_round_robin_size=4,
        # A fraction of the 1200 starting rating, not a rating. The old 8.0 stood for
        # nine thousand six hundred points of movement in one round.
        score_movement=0.008,
        ranking_stable_rounds=2,
        converged=True,
    )


def _artifact(stage: str, agent: str, schema_name: str, payload: dict, **kwargs):
    artifact = Artifact(
        stage=stage,
        agent=agent,
        artifact_type="specialist_output",
        content="",
        schema_name=schema_name,
        payload=payload,
        **kwargs,
    )
    return artifact


@pytest.fixture
def rich_session() -> Session:
    candidates = _candidates()
    plan = ResearchPlan(
        question="Can a protective coating improve lithium-ion battery cycle life?",
        intended_claim="hypothesis",
        constraints=["No cell may be cycled outside its rated voltage window"],
        assumptions=["Coating thickness is uniform to within ten per cent"],
        success_criteria=["A mechanism is distinguished from its leading rival"],
        stopping_criteria=["Two consecutive rounds without a ranking change"],
        governance_requirements=["Battery safety review before any cell is built"],
    )
    discovery = DiscoveryManifest(
        question=plan.question,
        source_leads=[
            SourceLead(canonical_url=_url(index), title=title, year=2024)
            for index, (title, _status, _confidence) in enumerate(_LEADS)
        ],
        narratives=[
            DiscoveryNarrative(
                question=plan.question,
                summary="Coating research concentrates on interphase stabilisation.",
                statements=[
                    DiscoveryStatement(
                        text=text,
                        facet="mechanism" if index % 2 else "prior_art",
                        originating_pass=1 + index // 4,
                        source_urls=[_url(index)],
                        relation="contradicts" if index % 3 == 0 else "supports",
                    )
                    for index, text in enumerate(_STATEMENTS)
                ],
                research_directions=["Interphase stabilisation", "Binder chemistry"],
                uncertainties=["Whether thickness or chemistry dominates"],
            )
        ],
        convergence_reason="coverage_satisfied",
    )
    evidence = EvidencePacket(
        question=plan.question,
        sources=[
            SourceRecord(
                id=f"source_{index}",
                url=_url(index),
                title=title,
                verification_status=status,
            )
            for index, (title, status, _confidence) in enumerate(_LEADS)
        ],
        claims=[
            EvidenceClaim(
                id=f"claim_{index}",
                claim=_CLAIMS[index],
                source_id=f"source_{index}",
                verification_status=status,
                confidence=confidence,
                relation="contradicts" if index % 4 == 0 else "supports",
            )
            for index, (_title, status, confidence) in enumerate(_LEADS)
        ],
        limitations=["No cell was built for this analysis"],
    )
    population = CandidatePopulation(
        candidates=candidates,
        comparison_criteria=["Mechanistic discrimination", "Cost of the decisive test"],
    )
    session = Session(
        question=plan.question,
        research_mode="experimental",
        approval_profile="auto",
    )
    session.artifacts = [
        _artifact("scope", "scope_planner", "ResearchPlan", plan.model_dump()),
        # "evidence", not "discovery": discovery is an agent, not a workflow stage, and
        # the orchestrator files its manifest under the evidence stage. A fixture stage
        # outside ``STAGES`` is a stage the report can only sort to the end.
        _artifact(
            "evidence", "discovery_agent", "DiscoveryManifest", discovery.model_dump()
        ),
        _artifact(
            "evidence",
            "evidence_agent",
            "EvidencePacket",
            evidence.model_dump(),
            payload_source="repaired",
            payload_repairs=["coerced confidence to a float"],
        ),
        _artifact(
            "generate",
            "generation_agent",
            "CandidatePopulation",
            population.model_dump(),
        ),
        _artifact(
            "reflect",
            "reflection_agent",
            "ReviewSet",
            _reviews(candidates).model_dump(),
            payload_source="deterministic_fallback",
            payload_error="the specialist answer failed contract validation",
        ),
        _artifact(
            "rank",
            "ranking_agent",
            "TournamentState",
            _tournament(candidates).model_dump(),
        ),
    ]
    return session
