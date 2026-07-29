from __future__ import annotations

import argparse

from .agents import A2AProvider, DeterministicProvider
from .dossier import write_dossier
from .ledger import ResearchLedger
from .models import RESEARCH_MODES, ApprovalMode, ApprovalProfile
from .orchestration import CoScientistWorkflow
from .tui import run_tui


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Human-steered multi-agent scientific co-scientist"
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "tui"):
        cmd = sub.add_parser(name)
        cmd.add_argument("question", nargs="?", help="Research question")
        cmd.add_argument("--resume", help="Load a saved session")
        cmd.add_argument("--save", help="Save session JSON")
        cmd.add_argument("--report", help="Write Markdown report")
        cmd.add_argument(
            "--auto",
            action="store_true",
            help="Alias for --approval-mode auto (demos/evaluation only)",
        )
        cmd.add_argument(
            "--approval-mode",
            choices=[mode.value for mode in ApprovalMode],
            help="Legacy alias: human maps to stage; auto maps to auto",
        )
        cmd.add_argument(
            "--approval-profile",
            choices=[profile.value for profile in ApprovalProfile],
            help=("Interaction policy: auto, milestone (default), stage, or artifact"),
        )
        cmd.add_argument(
            "--literature-only",
            action="store_true",
            help="Explicitly accept literature-only fallback for eligible missing inputs",
        )
        cmd.add_argument(
            "--input",
            action="append",
            default=[],
            metavar="TYPE=REFERENCE",
            help="Resolve a required scientific input by type (repeatable)",
        )
        cmd.add_argument("--db", help="SQLite research ledger path")
        cmd.add_argument("--session-id", help="Resume a session from --db")
        cmd.add_argument(
            "--research-mode",
            choices=RESEARCH_MODES,
            help="Override automatic scientific-method classification",
        )
        cmd.add_argument(
            "--provider",
            choices=("offline", "a2a"),
            default="offline",
            help="Specialist execution provider (default: offline)",
        )
        cmd.add_argument(
            "--a2a-url",
            default="http://127.0.0.1:8000",
            help="Base URL used with --provider a2a",
        )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    ledger = ResearchLedger(args.db) if args.db else None
    provider = (
        A2AProvider(args.a2a_url) if args.provider == "a2a" else DeterministicProvider()
    )
    if args.auto and args.approval_profile not in {None, ApprovalProfile.AUTO}:
        raise SystemExit("--auto conflicts with a non-auto --approval-profile.")
    requested_profile = (
        ApprovalProfile.AUTO
        if args.auto
        else ApprovalProfile(args.approval_profile)
        if args.approval_profile
        else ApprovalProfile.AUTO
        if args.approval_mode == ApprovalMode.AUTO
        else ApprovalProfile.STAGE
        if args.approval_mode == ApprovalMode.HUMAN
        else None
    )
    if args.session_id:
        if ledger is None:
            raise SystemExit("--session-id requires --db.")
        flow = CoScientistWorkflow.load_from_ledger(
            args.session_id, ledger, provider=provider
        )
        if requested_profile is not None:
            flow.session.approval_profile = requested_profile
            flow.session.approval_mode = (
                ApprovalMode.AUTO
                if requested_profile == ApprovalProfile.AUTO
                else ApprovalMode.HUMAN
            )
    elif args.resume:
        flow = CoScientistWorkflow.load(
            args.resume,
            provider=provider,
            ledger=ledger,
            approval_profile=requested_profile,
        )
    elif args.question:
        flow = CoScientistWorkflow(
            args.question,
            provider=provider,
            approval_profile=requested_profile or ApprovalProfile.MILESTONE,
            research_mode=args.research_mode,
            ledger=ledger,
        )
    else:
        raise SystemExit("Provide a question, --resume, or --session-id.")
    for item in args.input:
        input_type, separator, reference = item.partition("=")
        if not separator:
            raise SystemExit("--input must use TYPE=REFERENCE.")
        flow.provide_input(input_type, reference, actor="cli_researcher")
    if args.literature_only:
        flow.accept_literature_only(actor="cli_researcher")
    if flow.approval_profile != ApprovalProfile.AUTO:
        run_tui(flow)
    else:
        try:
            flow.run_auto()
        except ValueError as exc:
            if flow.session.status != "input_required":
                raise
            print(f"Input required: {exc}")
    if args.save:
        flow.save(args.save)
    if args.report:
        write_dossier(args.report, flow.render_report())
    print(
        f"Session {flow.session.id}: {flow.session.status}; "
        f"stage={flow.stage}; approval={flow.approval_profile.value}"
    )
    if flow.session.status == "input_required":
        return 2
    if flow.session.status == "evidence_required":
        return 4
    if flow.session.status == "governance_blocked":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
