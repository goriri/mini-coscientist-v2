import json
from pathlib import Path

from coscientist.cli import main


def test_cli_auto_mode_stops_at_mandatory_evidence_gate(tmp_path: Path):
    session_path = tmp_path / "auto-session.json"
    report_path = tmp_path / "auto-report.md"
    result = main(
        [
            "run",
            "Can a coating improve cycle life?",
            "--auto",
            "--save",
            str(session_path),
            "--report",
            str(report_path),
        ]
    )
    payload = json.loads(session_path.read_text())
    assert result == 4
    assert payload["status"] == "evidence_required"
    assert payload["approval_mode"] == "auto"
    assert all(
        decision["automatic"]
        for decision in payload["decisions"]
        if decision["action"] == "accept"
    )
    # A reader has to be able to tell, from the report alone, that no human signed
    # off on any gate. The regime is named in the goal's attributes and the caveat
    # is restated in the narrative.
    report = report_path.read_text()
    assert "Approval profile: auto" in report
    assert "Auto approval is a workflow convenience" in report


def test_the_evidence_halt_tells_the_operator_why_and_how_to_answer_it(
    tmp_path: Path, capsys
):
    """It printed a four-word status line and nothing else. The governance halt
    prints the finding and the exact command that clears it; a run that stops
    before generating a single hypothesis is no less recoverable, and the reason
    it stopped was recorded only in the event log."""
    session_path = tmp_path / "gate-session.json"
    result = main(
        [
            "run",
            "Can a coating improve cycle life?",
            "--auto",
            "--save",
            str(session_path),
        ]
    )
    output = capsys.readouterr().out

    assert result == 4
    assert "EVIDENCE GATE" in output
    # Which of the two checks failed, not merely that one did.
    assert "Discovery did not complete" in output
    assert "verified against the" in output
    # The remedy, spelled out against this run's own session file.
    assert f"--resume {session_path} --allow-exploratory-evidence" in output
    assert "unverified hypothesis" in output


def test_cli_auto_mode_completes_only_with_explicit_exploratory_opt_in(
    tmp_path: Path, capsys
):
    session_path = tmp_path / "exploratory-session.json"
    report_path = tmp_path / "exploratory-report.md"
    result = main(
        [
            "run",
            "Can a coating improve cycle life?",
            "--auto",
            "--allow-exploratory-evidence",
            "--save",
            str(session_path),
            "--report",
            str(report_path),
        ]
    )
    payload = json.loads(session_path.read_text())
    output = capsys.readouterr().out

    assert result == 0
    assert payload["status"] != "evidence_required"
    assert payload["exploratory_evidence_accepted"] is True
    assert any(
        event["event_type"] == "limited_exploratory_evidence_accepted"
        for event in payload["events"]
    )
    assert "UNVERIFIED" in output
    # The dossier must never present the waived material as verified findings.
    assert "not verified findings" in report_path.read_text()


def test_tui_default_milestone_mode_requires_four_accept_inputs(monkeypatch, capsys):
    actions = iter(["a", "x", "a", "a", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(actions))
    result = main(["tui", "Can a coating improve cycle life?"])
    output = capsys.readouterr().out
    assert result == 0
    assert output.count("SUPERVISOR BUNDLE") == 4
    assert "All stages accepted" in output
    assert "approval=milestone" in output


UNSAFE_QUESTION = (
    "Should the agent begin carrying out the electrode protocol in a hazardous "
    "laboratory without human safety review?"
)


def _run_to_governance_block(session_path: Path, capsys) -> str:
    """Drive the CLI to a real governance halt and return what it printed."""
    result = main(
        [
            "run",
            UNSAFE_QUESTION,
            "--auto",
            "--allow-exploratory-evidence",
            "--save",
            str(session_path),
        ]
    )
    assert result == 3
    assert json.loads(session_path.read_text())["status"] == "governance_blocked"
    return capsys.readouterr().out


def test_a_governance_halt_tells_the_operator_how_to_answer_it(tmp_path: Path, capsys):
    """A gate nobody can clear is a gate that gets deleted instead."""
    session_path = tmp_path / "blocked-session.json"
    output = _run_to_governance_block(session_path, capsys)
    assert "GOVERNANCE BLOCK" in output
    assert "--adjudicate-governance" in output
    assert "--adjudicator" in output
    # The flaw itself, not just the fact that there was one.
    assert "unsafe real-world autonomy" in output


def test_an_unattributed_override_is_refused_before_anything_is_recorded(
    tmp_path: Path, capsys
):
    session_path = tmp_path / "blocked-session.json"
    _run_to_governance_block(session_path, capsys)
    blocked = json.loads(session_path.read_text())
    review_id = blocked["events"][-1]["payload"]["review_ids"][0]

    try:
        main(
            [
                "run",
                "--resume",
                str(session_path),
                "--auto",
                "--adjudicate-governance",
                f"{review_id}=override:Looks fine to me.",
                "--save",
                str(session_path),
            ]
        )
    except SystemExit as exit_error:
        assert "requires --adjudicator" in str(exit_error)
    else:  # pragma: no cover - the call must not succeed
        raise AssertionError("An anonymous safety override was accepted.")
    # The refusal happens before the session is touched.
    assert json.loads(session_path.read_text())["governance_adjudications"] == []


def test_a_named_withdrawal_clears_the_block_and_finishes_the_run(
    tmp_path: Path, capsys
):
    session_path = tmp_path / "blocked-session.json"
    _run_to_governance_block(session_path, capsys)
    blocked = json.loads(session_path.read_text())
    review_ids = blocked["events"][-1]["payload"]["review_ids"]
    before = len(
        next(
            artifact
            for artifact in reversed(blocked["artifacts"])
            if artifact["schema_name"] == "CandidatePopulation"
        )["payload"]["candidates"]
    )

    result = main(
        [
            "run",
            "--resume",
            str(session_path),
            "--auto",
            "--adjudicator",
            "R. Safety",
            *[
                argument
                for review_id in review_ids[:1]
                for argument in (
                    "--adjudicate-governance",
                    f"{review_id}=withdraw:The protocol cannot be run without "
                    "qualified supervision, so the hypothesis is dropped.",
                )
            ],
            *[
                argument
                for review_id in review_ids[1:]
                for argument in (
                    "--adjudicate-governance",
                    f"{review_id}=override:Accepted; the remaining hypotheses "
                    "are simulation-only and touch no bench protocol.",
                )
            ],
            "--save",
            str(session_path),
        ]
    )

    payload = json.loads(session_path.read_text())
    assert result == 0
    assert payload["status"] == "ready_for_report"
    adjudications = payload["governance_adjudications"]
    assert len(adjudications) == len(review_ids)
    assert all(item["adjudicator"] == "R. Safety" for item in adjudications)
    assert all(item["justification"].strip() for item in adjudications)
    # The withdrawn hypothesis is gone from the population the tournament saw,
    # and the population it was reviewed against is still on file.
    surviving = next(
        artifact
        for artifact in reversed(payload["artifacts"])
        if artifact["schema_name"] == "CandidatePopulation"
    )["payload"]["candidates"]
    assert len(surviving) == before - 1
    superseded = [
        artifact
        for artifact in payload["artifacts"]
        if artifact["schema_name"] == "CandidatePopulation"
        and artifact["status"] == "superseded"
    ]
    assert superseded, "the reviewed population must remain readable"


DATA_QUESTION = (
    "Analyze scRNA-seq profiles to identify the cell clusters that respond to "
    "a protective interphase coating."
)


def test_a_fresh_run_stops_at_the_input_gate_when_nothing_waives_it(tmp_path: Path):
    """The baseline the flag below is measured against."""
    session_path = tmp_path / "gated.json"
    result = main(["run", DATA_QUESTION, "--auto", "--save", str(session_path)])

    assert result == 2
    assert json.loads(session_path.read_text())["status"] == "input_required"


def test_literature_only_is_answered_at_the_gate_and_not_before_the_run(
    tmp_path: Path,
):
    """It used to be applied before the first stage, where a fresh session has no
    input requirements at all -- the scope stage is what writes them. So the flag
    that exists to get a run past the input gate aborted it with a traceback
    several stages earlier, and worked only on a session resumed at the gate."""
    session_path = tmp_path / "waived.json"
    result = main(
        [
            "run",
            DATA_QUESTION,
            "--auto",
            "--literature-only",
            "--save",
            str(session_path),
        ]
    )
    payload = json.loads(session_path.read_text())

    # Past scope, and stopped at the next gate rather than at this one.
    assert result == 4
    assert payload["status"] == "evidence_required"
    assert payload["literature_only"] is True
    assert [requirement["status"] for requirement in payload["input_requirements"]] == [
        "fallback_accepted"
    ]


def test_both_waivers_together_carry_one_run_past_both_gates(tmp_path: Path):
    """Either gate can be the first one reached and answering one exposes the
    other, so the two flags are answered in one loop rather than in a fixed
    order. Each is offered once: a flag that re-answered its own gate would turn
    a run that cannot pass it into a run that never stops."""
    session_path = tmp_path / "both.json"
    result = main(
        [
            "run",
            DATA_QUESTION,
            "--auto",
            "--literature-only",
            "--allow-exploratory-evidence",
            "--save",
            str(session_path),
        ]
    )
    payload = json.loads(session_path.read_text())

    assert result == 0
    assert payload["status"] == "ready_for_report"
    assert payload["literature_only"] is True
    assert payload["exploratory_evidence_accepted"] is True
