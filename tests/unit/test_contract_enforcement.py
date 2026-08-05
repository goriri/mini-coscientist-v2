"""A live specialist's unusable output must stop the run, never be replaced."""

import json

import pytest

from coscientist.agents import (
    SPECIALISTS,
    ContractViolation,
    Specialist,
    output_contract,
)
from coscientist.contract_io import parse_contract, schema_instruction
from coscientist.models import ApprovalProfile, ReviewSet
from coscientist.orchestration import CoScientistWorkflow
from coscientist.parity import ROLE_CONTRACTS


class _ScriptedProvider:
    """A live-looking provider that returns whatever the test scripted."""

    model_id = "scripted-test-model"

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, *, role: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


def _scoped_session():
    flow = CoScientistWorkflow(
        "Can a protective coating improve battery cycle life?",
        approval_profile=ApprovalProfile.AUTO,
    )
    flow.accept(flow.preview(), automatic=True)
    return flow.session


def _specialist(role: str) -> Specialist:
    return next(item for item in SPECIALISTS if item.role == role)


def test_unparseable_live_output_halts_instead_of_substituting_a_template():
    session = _scoped_session()
    provider = _ScriptedProvider("I would rather describe my approach in prose.")
    with pytest.raises(ContractViolation) as raised:
        _specialist("generation").run(session, provider)
    assert raised.value.role == "generation"
    assert raised.value.error
    # The repair round-trip is the whole reason a second call is worth making.
    assert len(provider.prompts) == 2
    assert "could not be parsed" in provider.prompts[1]
    assert "CandidatePopulation" in provider.prompts[1]


def test_a_repaired_second_attempt_is_kept_and_recorded():
    session = _scoped_session()
    # "category" and "prediction" are the field names models actually emit for
    # CandidatePopulation's generation_strategy and predictions.
    payload = {
        "candidates": [
            {
                "id": f"cand_{index}",
                "category": "evidence_first",
                "claim": f"Coating variant {index} extends cycle life.",
                "rationale": "Interface stabilisation is directly measurable.",
                "prediction": ["Capacity fade slows versus the uncoated cell."],
                "falsifier": "No difference in fade after 500 cycles.",
            }
            for index in range(2)
        ]
    }
    provider = _ScriptedProvider("prose only, no contract", json.dumps(payload))
    artifact = _specialist("generation").run(session, provider)
    assert artifact.schema_name == "CandidatePopulation"
    assert artifact.payload_source == "repaired"
    assert artifact.payload_repairs
    assert len(artifact.payload["candidates"]) == 2


def test_the_offline_provider_is_never_treated_as_a_contract_violation():
    from coscientist.agents import DeterministicProvider

    session = _scoped_session()
    artifact = _specialist("generation").run(session, DeterministicProvider())
    assert artifact.payload_source == "deterministic_fallback"
    assert artifact.payload["candidates"]


def test_every_contract_role_states_its_schema_in_the_prompt_it_receives():
    """The prose brief alone once named fields the contracts do not have."""
    for role, model in ROLE_CONTRACTS.items():
        if role in {"evidence_discovery", "source_verification"}:
            continue  # these stages run through the evidence controller
        instruction = output_contract(role)
        assert model.__name__ in instruction
        for name, info in model.model_fields.items():
            if info.is_required():
                assert name in instruction, f"{role}: {name} missing from contract"


def test_a_review_set_keeps_the_reviews_a_specialist_actually_wrote():
    schema = schema_instruction(ReviewSet)
    assert "reviews" in schema and "CandidateReview" in schema
    outcome = parse_contract(
        json.dumps(
            {
                "reviews": [
                    {
                        "candidate_id": "cand_0",
                        "reviewer": "novelty_review",
                        "findings": ["Prior art covers the coating chemistry."],
                        "score": 3,
                    }
                ]
            }
        ),
        ReviewSet,
    )
    assert outcome.ok
    review = outcome.value.reviews[0]
    # criterion and recommendation are required but routinely omitted; they are
    # derived rather than allowed to void an otherwise complete review.
    assert review.criterion == "novelty"
    assert review.recommendation in {
        "advance",
        "revise",
        "reject",
        "insufficient_evidence",
    }
    assert outcome.repairs
