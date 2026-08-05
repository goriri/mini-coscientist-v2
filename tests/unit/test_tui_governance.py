"""The interactive path out of a safety block.

The CLI flags have their own tests. This file covers the terminal loop, which
was the worse of the two failures: it ran ``while ... status == "active"``, so a
governance block simply ended the session, printing nothing about the fatal flaw
that stopped it. An operator would have seen a run that quietly finished early.
"""

from __future__ import annotations

import builtins

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.governance import latest_population, open_blockers
from coscientist.models import ApprovalProfile, CandidatePopulation
from coscientist.orchestration import CoScientistWorkflow
from coscientist.tui import run_tui

UNSAFE_QUESTION = (
    "Should the agent begin carrying out the electrode protocol in a hazardous "
    "laboratory without human safety review?"
)
# The flaw is a property of the question, so the offline reviewer flags the
# whole population. Every one of them has to be answered separately.
FLAGGED = 8
NAME = "Dr. Ada Lovelace"
REASON = "The protocol is unsafe as written."


class Operator:
    """A scripted human.

    Prompts are answered by what they ask rather than by position, so a test
    says only what it is about: one decision per governance finding, in order.
    Everything else -- accepting a stage, waiving the evidence gate the offline
    provider can never satisfy -- is the uninteresting background of getting
    there.
    """

    def __init__(self, decisions: list[str], *, name: str = NAME, reason: str = REASON):
        self.decisions = list(decisions)
        self.names = [name]
        self.reasons = [reason]

    def __call__(self, prompt: str) -> str:
        if "[w]ithdraw" in prompt:
            if not self.decisions:
                raise AssertionError("more governance findings than decisions")
            return self.decisions.pop(0)
        if prompt.startswith("Your name"):
            return self.names.pop(0) if len(self.names) > 1 else self.names[0]
        if prompt.startswith("Reason"):
            return self.reasons.pop(0) if len(self.reasons) > 1 else self.reasons[0]
        if "[r]etry" in prompt:
            return "x"
        if "[a]ccept" in prompt:
            return "a"
        raise AssertionError(f"The TUI asked something unscripted: {prompt!r}")


@pytest.fixture
def operator(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted human and keep a record of what it was asked."""

    def _install(responder: Operator) -> list[str]:
        prompts: list[str] = []

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            return responder(prompt)

        monkeypatch.setattr(builtins, "input", fake_input)
        return prompts

    return _install


def _blocked_workflow() -> CoScientistWorkflow:
    return CoScientistWorkflow(
        UNSAFE_QUESTION,
        DeterministicProvider(),
        approval_profile=ApprovalProfile.MILESTONE,
    )


def _survivors(flow: CoScientistWorkflow) -> set[str]:
    population = latest_population(flow.session)
    return {
        item.id
        for item in CandidatePopulation.model_validate(population.payload).candidates
    }


def test_the_block_is_announced_with_the_flaw_that_caused_it(operator, capsys):
    flow = _blocked_workflow()
    operator(Operator(["s"]))

    run_tui(flow)

    output = capsys.readouterr().out
    assert "GOVERNANCE BLOCK" in output
    assert "FATAL:" in output
    assert "unsafe real-world autonomy" in output


def test_overriding_every_finding_clears_the_block_and_keeps_the_hypotheses(
    operator, capsys
):
    flow = _blocked_workflow()
    operator(Operator(["o"] * FLAGGED))

    run_tui(flow)

    output = capsys.readouterr().out
    assert "carries a fatal safety" in output
    assert flow.done
    assert not open_blockers(flow.session)
    decided = {item.candidate_id for item in flow.session.governance_adjudications}
    assert len(decided) == FLAGGED
    assert {item.resolution for item in flow.session.governance_adjudications} == {
        "override"
    }
    assert decided <= _survivors(flow), (
        "an override that removes the hypothesis is a withdrawal wearing its name"
    )


def test_withdrawing_drops_the_hypothesis_from_the_population(operator, capsys):
    flow = _blocked_workflow()
    operator(Operator(["w"] + ["o"] * (FLAGGED - 1)))

    run_tui(flow)

    withdrawn = next(
        item
        for item in flow.session.governance_adjudications
        if item.resolution == "withdraw"
    )
    assert withdrawn.adjudicator == NAME
    assert withdrawn.justification == REASON
    assert withdrawn.candidate_id not in _survivors(flow)
    assert len(_survivors(flow)) == FLAGGED - 1


def test_stopping_records_nothing_and_leaves_the_finding_open(operator, capsys):
    flow = _blocked_workflow()
    operator(Operator(["s"]))

    run_tui(flow)

    assert flow.session.status == "stopped_by_researcher"
    assert flow.session.governance_adjudications == []
    assert len(open_blockers(flow.session)) == FLAGGED
    assert "remains unanswered" in capsys.readouterr().out


def test_every_flagged_hypothesis_is_answered_on_its_own(operator, capsys):
    """One keystroke must not clear findings the operator never read."""
    flow = _blocked_workflow()
    operator(Operator(["w", "s"]))

    run_tui(flow)

    assert flow.session.status == "stopped_by_researcher"
    assert len(flow.session.governance_adjudications) == 1
    assert len(open_blockers(flow.session)) == FLAGGED - 1


def test_the_last_hypothesis_cannot_be_quietly_withdrawn(operator, capsys):
    """Emptying the population would leave a report about nothing."""
    flow = _blocked_workflow()
    operator(Operator(["w"] * FLAGGED + ["o"]))

    run_tui(flow)

    output = capsys.readouterr().out
    assert "Refused:" in output
    assert len(_survivors(flow)) == 1
    resolutions = [item.resolution for item in flow.session.governance_adjudications]
    assert resolutions == ["withdraw"] * (FLAGGED - 1) + ["override"]


def test_a_decision_is_not_recorded_without_a_name(operator, capsys):
    """Blank is not an answer; the prompt repeats rather than storing an empty one."""
    flow = _blocked_workflow()
    responder = Operator(["w"] + ["o"] * (FLAGGED - 1))
    responder.names = ["", "   ", NAME]
    responder.reasons = ["", REASON]
    operator(responder)

    run_tui(flow)

    assert "Required." in capsys.readouterr().out
    first = flow.session.governance_adjudications[0]
    assert first.adjudicator == NAME
    assert first.justification == REASON


def test_an_unrecognized_key_is_not_treated_as_a_decision(operator, capsys):
    flow = _blocked_workflow()
    prompts = operator(Operator(["y", "w"] + ["o"] * (FLAGGED - 1)))

    run_tui(flow)

    assert "Choose w, o, or s." in capsys.readouterr().out
    assert flow.session.governance_adjudications[0].resolution == "withdraw"
    assert sum("[w]ithdraw" in prompt for prompt in prompts) == FLAGGED + 1


def test_the_run_continues_to_completion_once_the_block_is_answered(operator, capsys):
    flow = _blocked_workflow()
    operator(Operator(["w"] + ["o"] * (FLAGGED - 1)))

    run_tui(flow)

    assert flow.done
    assert "All stages accepted" in capsys.readouterr().out
