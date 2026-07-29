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
