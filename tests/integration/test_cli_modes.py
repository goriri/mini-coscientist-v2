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
    assert "**Approval policy:** auto" in report_path.read_text()


def test_tui_default_milestone_mode_requires_four_accept_inputs(monkeypatch, capsys):
    actions = iter(["a", "x", "a", "a", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(actions))
    result = main(["tui", "Can a coating improve cycle life?"])
    output = capsys.readouterr().out
    assert result == 0
    assert output.count("SUPERVISOR BUNDLE") == 4
    assert "All stages accepted" in output
    assert "approval=milestone" in output
