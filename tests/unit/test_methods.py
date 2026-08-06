import json

import pytest

from coscientist.agents import CRITIC_ROUNDS, Specialist
from coscientist.methods import classify_research_mode, method_requirements
from coscientist.models import RESEARCH_MODES, Session
from coscientist.orchestration import CoScientistWorkflow


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Test a coating in a controlled battery experiment", "experimental"),
        ("Analyze an observational cohort for risk factors", "observational"),
        ("Benchmark a machine learning algorithm on a dataset", "computational"),
        ("Prove a theorem and verify it with simulation", "theory_simulation"),
        ("Conduct a systematic review and meta-analysis", "systematic_review"),
        ("Calibrate a field sensor and its uncertainty", "measurement_field"),
    ],
)
def test_domain_general_mode_classification(question: str, expected: str):
    assert classify_research_mode(question) == expected
    flow = CoScientistWorkflow(question)
    assert flow.session.research_mode == expected
    assert method_requirements(expected)


def test_every_declared_mode_has_a_method_adapter():
    assert all(method_requirements(mode) for mode in RESEARCH_MODES)


class _MockActorCriticProvider:
    """An actor whose drafts satisfy the contract, so the critic is what is tested.

    A mock that returns prose is stopped by the contract halt long before the
    critic sees anything, which makes the round count a fact about the parser
    rather than about the loop.
    """

    def __init__(self, satisfy_on_round: int = 2):
        self.satisfy_on_round = satisfy_on_round
        self.calls = []
        self.model_id = "mock-actor-critic"

    def complete(self, *, role: str, prompt: str) -> str:
        self.calls.append(role)
        if role.endswith("_critic"):
            critic_rounds = sum(1 for call in self.calls if call.endswith("_critic"))
            if critic_rounds >= self.satisfy_on_round:
                return "SATISFIED"
            return "Please add more controls and domain falsifiers."
        drafts = sum(1 for call in self.calls if not call.endswith("_critic"))
        return json.dumps(
            {
                "research_mode": "experimental",
                "question": "Test question",
                "intended_claim": f"Research objective: revision {drafts}",
                "success_criteria": ["A measurable effect against a baseline."],
            }
        )


def test_the_critic_loop_stops_the_round_it_is_satisfied():
    session = Session(question="Test question")
    specialist = Specialist("scope", "goal_manager", "Manage research goals")
    provider = _MockActorCriticProvider(satisfy_on_round=2)

    artifact = specialist.run(session, provider)

    assert [call for call in provider.calls if call.endswith("_critic")] == [
        "goal_manager_critic"
    ] * 2
    # One opening draft and one revision: the satisfied round asks for nothing.
    assert [call for call in provider.calls if not call.endswith("_critic")] == [
        "goal_manager"
    ] * 2
    assert artifact.payload["intended_claim"] == "Research objective: revision 2"


def test_a_critic_that_never_relents_is_stopped_by_the_round_bound():
    """Two rounds is the bound. Without it a run makes calls nobody waits for."""
    session = Session(question="Test question")
    specialist = Specialist("scope", "goal_manager", "Manage research goals")
    provider = _MockActorCriticProvider(satisfy_on_round=99)

    specialist.run(session, provider)

    critics = [call for call in provider.calls if call.endswith("_critic")]
    assert len(critics) == CRITIC_ROUNDS
