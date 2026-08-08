from __future__ import annotations

import argparse

from .agents import A2AProvider, ContractViolation, DeterministicProvider
from .dossier import write_dossier
from .governance import open_blockers
from .ledger import ResearchLedger
from .model_catalog import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    LANGUAGE_CODES,
    MODEL_IDS,
)
from .models import RESEARCH_MODES, ApprovalMode, ApprovalProfile
from .orchestration import CoScientistWorkflow
from .parity import unresolved_blockers
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
            "--allow-exploratory-evidence",
            action="store_true",
            help=(
                "Non-interactive equivalent of the TUI exploratory fallback: "
                "continue past the evidence gate with unverified material. Every "
                "downstream output stays a hypothesis, never an evidence-backed "
                "finding."
            ),
        )
        cmd.add_argument(
            "--input",
            action="append",
            default=[],
            metavar="TYPE=REFERENCE",
            help="Resolve a required scientific input by type (repeatable)",
        )
        cmd.add_argument(
            "--adjudicate-governance",
            action="append",
            default=[],
            metavar="REVIEW_ID=RESOLUTION:JUSTIFICATION",
            help=(
                "Answer one fatal governance finding on a blocked session "
                "(repeatable). RESOLUTION is 'withdraw' to drop the hypothesis "
                "or 'override' to keep it and accept the flaw. The justification "
                "is required and is reprinted in the dossier beside the flaw."
            ),
        )
        cmd.add_argument(
            "--adjudicator",
            help=(
                "Name of the person accountable for --adjudicate-governance "
                "decisions. Required whenever one is given."
            ),
        )
        cmd.add_argument(
            "--evidence-review",
            action="store_true",
            help=(
                "Stop after discovery so the evidence base can be read before "
                "the generators reason over it. The milestone profile does not "
                "gate evidence otherwise; --auto ignores this, having nobody to "
                "ask. Set once for a new run and kept for its whole life."
            ),
        )
        cmd.add_argument("--db", help="SQLite research ledger path")
        cmd.add_argument("--session-id", help="Resume a session from --db")
        cmd.add_argument(
            "--research-mode",
            choices=RESEARCH_MODES,
            help="Override automatic scientific-method classification",
        )
        cmd.add_argument(
            "--model",
            choices=MODEL_IDS,
            help=(
                "Reasoning model for a new run (default: "
                f"{DEFAULT_MODEL}). A resumed run keeps the model it started "
                "on, so this is rejected with --resume or --session-id."
            ),
        )
        cmd.add_argument(
            "--language",
            choices=LANGUAGE_CODES,
            help=(
                "Language the specialists write in (default: "
                f"{DEFAULT_LANGUAGE}). The report's own headings and analysis "
                "stay in English; this governs the scientific content."
            ),
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
    # A run being resumed carries its own model, and the workflow refuses a
    # conflicting one, so the flag is only ever a choice for a new run. Passing
    # its default through on a resume would reject every session that had been
    # started on anything else.
    resuming = bool(args.resume or args.session_id)
    if resuming and (args.model or args.language or args.evidence_review):
        raise SystemExit(
            "--model, --language and --evidence-review configure a new run. A "
            "resumed session keeps the model, language and gates it was "
            "started with."
        )
    provider = (
        A2AProvider(args.a2a_url, model=args.model or DEFAULT_MODEL)
        if args.provider == "a2a"
        else DeterministicProvider()
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
            model=args.model or DEFAULT_MODEL,
            language=args.language or DEFAULT_LANGUAGE,
            evidence_review=args.evidence_review,
            ledger=ledger,
        )
    else:
        raise SystemExit("Provide a question, --resume, or --session-id.")
    for item in args.input:
        input_type, separator, reference = item.partition("=")
        if not separator:
            raise SystemExit("--input must use TYPE=REFERENCE.")
        flow.provide_input(input_type, reference, actor="cli_researcher")
    if args.literature_only and unresolved_blockers(flow.session):
        # A resumed session is already parked at the input gate, so the fallback
        # can be accepted before anything runs. A fresh one has no requirements
        # yet -- the scope stage is what writes them -- and accepting one here
        # aborted the run with "no unresolved input supports a literature-only
        # fallback" before its first stage. A fresh run reaches the gate below.
        flow.accept_literature_only(actor="cli_researcher")
    for item in args.adjudicate_governance:
        review_id, separator, verdict = item.partition("=")
        resolution, colon, justification = verdict.partition(":")
        if not separator or not colon:
            raise SystemExit(
                "--adjudicate-governance must use REVIEW_ID=RESOLUTION:JUSTIFICATION."
            )
        if not args.adjudicator:
            # An unattributed safety override is the failure this whole path
            # exists to prevent, so it is refused before anything is recorded.
            raise SystemExit(
                "--adjudicate-governance requires --adjudicator naming the "
                "person accountable for the decision."
            )
        flow.adjudicate_governance(
            review_id.strip(),
            resolution.strip(),
            adjudicator=args.adjudicator,
            justification=justification.strip(),
        )
    if flow.approval_profile != ApprovalProfile.AUTO:
        run_tui(flow)
    else:
        # Two gates halt an unattended run -- a missing scientific input, and
        # evidence nothing could verify -- and each has a flag that says in
        # advance how it is to be answered. Either can be the first one reached,
        # and answering one exposes the other, so they are answered here in one
        # loop rather than in a fixed order. Each waiver is offered once: a flag
        # that re-answered its own gate would turn a run that cannot get past it
        # into a run that never stops.
        offer_literature_only = args.literature_only
        offer_exploratory = args.allow_exploratory_evidence
        while True:
            try:
                flow.run_auto()
            except ValueError as exc:
                if flow.session.status != "input_required":
                    raise
                if not offer_literature_only:
                    print(f"Input required: {exc}")
                    break
                offer_literature_only = False
                flow.accept_literature_only(actor="cli_researcher")
                continue
            except ContractViolation as exc:
                # A live specialist produced output the contract rejects.
                # Substituting a template here would put words in the model's
                # mouth, so the run stops and hands the failure to a human with
                # the raw response saved.
                print(
                    f"\nStage halted: {exc}\n"
                    f"The unusable response was saved so it can be inspected.\n"
                )
                if args.save:
                    flow.save(args.save)
                print(
                    f"Session {flow.session.id}: contract_violation; stage={flow.stage}"
                )
                return 5
            if flow.session.status == "evidence_required" and offer_exploratory:
                # Mirror the TUI's e[x]ploratory branch: waive the
                # verified-evidence gate once, explicitly, then let the run
                # continue. The waiver is recorded in the session ledger and
                # never inferred automatically.
                offer_exploratory = False
                print(
                    "\nWARNING: --allow-exploratory-evidence accepted. Evidence "
                    "could not be verified; every downstream candidate, review, "
                    "and recommendation is an UNVERIFIED hypothesis.\n"
                )
                flow.accept_exploratory_evidence(actor="cli_researcher")
                pending = flow.pending_draft
                if pending is not None:
                    flow.accept(pending, actor="cli_researcher")
                continue
            break
    if args.save:
        flow.save(args.save)
    if args.report:
        write_dossier(args.report, flow.render_report())
    if flow.session.status == "governance_blocked":
        # A halt a human cannot act on is a dead end. Print what was found, and
        # the exact command that answers it, next to the reason it stopped.
        print("\nGOVERNANCE BLOCK. The safety and governance reviewer recorded a")
        print("fatal flaw. Nothing advances until a named person answers it.\n")
        for blocker in open_blockers(flow.session):
            print(f"  {blocker.review_id}  (hypothesis {blocker.candidate_id})")
            for flaw in blocker.review.fatal_flaws:
                print(f"    - {flaw}")
        example = open_blockers(flow.session)
        if example and args.save:
            print(
                f"\nResolve with:\n"
                f"  --resume {args.save} --adjudicator 'Your Name' \\\n"
                f"    --adjudicate-governance "
                f"'{example[0].review_id}=withdraw:why this is the right call'\n"
                f"Use 'override' instead of 'withdraw' to keep the hypothesis "
                f"and accept the flaw on the record.\n"
            )
    if flow.session.status == "evidence_required":
        # The governance gate below already prints what it found and the command
        # that answers it. This gate used to print nothing at all: a run stopped
        # before generating a single hypothesis and said so in four words on the
        # status line, leaving the reason and the remedy for the reader to find in
        # the event log. The two halts are equally recoverable, so they say so
        # equally.
        diagnosis = next(
            (
                event.payload
                for event in reversed(flow.session.events)
                if event.event_type == "evidence_verification_required"
            ),
            {},
        )
        print("\nEVIDENCE GATE. Generation is blocked because the literature this")
        print("run retrieved was not verified. Nothing downstream would be a")
        print("finding, so nothing downstream was produced.\n")
        if not diagnosis.get("manifest_ok", True):
            print("  - Discovery did not complete: no source lead was returned, or")
            print("    a search run failed, timed out, or came back incomplete.")
        if not diagnosis.get("packet_ok", True):
            print("  - No claim in the evidence packet was verified against the")
            print("    source it cites.")
        print("\nEither retry discovery, or accept the limited exploratory workflow:")
        print(
            f"  --resume {args.save} --allow-exploratory-evidence"
            if args.save
            else "  re-run with --save SESSION.json --allow-exploratory-evidence"
        )
        print(
            "Under that waiver every candidate, review, and recommendation stays "
            "an\nunverified hypothesis, and the report says so on its face.\n"
        )
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
