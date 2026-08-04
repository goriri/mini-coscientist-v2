import pytest

from coscientist.methods import classify_research_mode, method_requirements
from coscientist.models import RESEARCH_MODES
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
    def __init__(self, satisfy_on_round: int = 2):
        self.satisfy_on_round = satisfy_on_round
        self.calls = []
        self.model_id = "mock-actor-critic"

    def complete(self, *, role: str, prompt: str) -> str:
        self.calls.append(role)
        if role.endswith("_critic"):
            critic_rounds = sum(1 for c in self.calls if c.endswith("_critic"))
            if critic_rounds >= self.satisfy_on_round:
                return "SATISFIED"
            return "Please add more controls and domain falsifiers."
        return "Research objective: Test question"


def test_specialist_actor_critic_loop_reaches_satisfied():
    from coscientist.agents import Specialist
    from coscientist.models import Session
    session = Session(question="Test question")
    specialist = Specialist("scope", "goal_manager", "Manage research goals")
    provider = _MockActorCriticProvider(satisfy_on_round=3)
    artifact = specialist.run(session, provider)
    assert "Research objective:" in artifact.content
    critic_calls = [c for c in provider.calls if c.endswith("_critic")]
    assert len(critic_calls) == 3

