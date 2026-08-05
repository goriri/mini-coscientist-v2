from __future__ import annotations

from .governance import WithdrawalRefused, open_blockers
from .models import ApprovalProfile
from .orchestration import CoScientistWorkflow


def _required(prompt: str) -> str:
    """Read a value the record cannot be written without."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Required. A safety decision is not recorded without it.")


def _adjudicate(workflow: CoScientistWorkflow) -> bool:
    """Walk a human through every open governance finding.

    Returns ``False`` when the operator stops instead of answering. Each finding
    is presented and decided on its own: one blanket confirmation covering
    several unrelated hazards would defeat the point of asking.
    """
    print(
        "\nGOVERNANCE BLOCK. The safety and governance reviewer recorded a fatal\n"
        "flaw. Nothing advances until you answer it. Your name and your reason\n"
        "are recorded and reprinted in the dossier beside the flaw."
    )
    for blocker in list(open_blockers(workflow.session)):
        print(f"\n{'-' * 72}\nHYPOTHESIS: {blocker.candidate_id}")
        for flaw in blocker.review.fatal_flaws:
            print(f"  FATAL: {flaw}")
        for objection in blocker.review.objections:
            print(f"  objection: {objection}")
        while True:
            action = (
                input("\n[w]ithdraw hypothesis, [o]verride and accept, [s]top: ")
                .strip()
                .lower()
            )
            if action in {"s", "stop"}:
                workflow.stop()
                print("Session stopped; the finding remains unanswered on the record.")
                return False
            if action not in {"w", "withdraw", "o", "override"}:
                print("Choose w, o, or s.")
                continue
            resolution = "withdraw" if action.startswith("w") else "override"
            if resolution == "override":
                print(
                    "Overriding keeps a hypothesis that carries a fatal safety\n"
                    "finding. It stays in the report, and so does this decision."
                )
            adjudicator = _required("Your name: ")
            justification = _required("Reason (recorded verbatim): ")
            try:
                workflow.adjudicate_governance(
                    blocker.review_id,
                    resolution,
                    adjudicator=adjudicator,
                    justification=justification,
                )
            except WithdrawalRefused as refusal:
                print(f"Refused: {refusal}")
                continue
            break
    return True


def run_tui(workflow: CoScientistWorkflow) -> None:
    """Portable line-oriented TUI; works over SSH and needs no UI dependency."""
    print("\nCo-Scientist — selectable human-in-the-loop research workflow")
    print(f"Approval profile: {workflow.approval_profile.value}")
    budget = workflow.session.budget
    print(
        "Budgets: "
        f"candidates={budget.max_candidates}, "
        f"comparisons={budget.max_pairwise_comparisons}, "
        f"searches={budget.max_searches}, "
        f"evolution_rounds={budget.max_evolution_rounds}"
    )
    print("Every output is a draft. Verify evidence before research action.\n")
    # A governance block is a state the loop has to stay awake for. It is set
    # inside accept(), so a loop that only runs while "active" would fall out
    # of the bottom and end the session without ever telling the operator that
    # a safety finding was what stopped it.
    while not workflow.done and workflow.session.status in {
        "active",
        "governance_blocked",
    }:
        workflow.advance_to_human_gate()
        if workflow.session.status == "governance_blocked":
            if not _adjudicate(workflow):
                return
            continue
        if workflow.session.status == "evidence_required":
            print(
                "\nEvidence verification is incomplete. Generation is blocked; "
                "all discovered material remains unverified."
            )
            action = (
                input("[r]etry Deep Research, e[x]ploratory fallback, [s]top: ")
                .strip()
                .lower()
            )
            if action in {"x", "exploratory"}:
                workflow.accept_exploratory_evidence(actor="cli_researcher")
                workflow.accept(workflow.pending_draft, actor="cli_researcher")
                continue
            if action in {"r", "retry"}:
                workflow.retry_evidence(actor="cli_researcher")
                continue
            workflow.stop()
            return
        if workflow.done or workflow.session.status != "active":
            break
        draft = workflow.preview()
        if workflow.approval_profile == ApprovalProfile.ARTIFACT:
            for specialist_artifact in list(workflow.pending_artifact_reviews):
                print(
                    f"\n{'=' * 72}\nARTIFACT: {specialist_artifact.agent.upper()}  |  "
                    f"SCHEMA: {specialist_artifact.schema_name}\n"
                    f"{'-' * 72}\n{specialist_artifact.content}\n"
                )
                action = input("[a]pprove artifact, [s]top: ").strip().lower()
                if action in {"s", "stop"}:
                    workflow.stop()
                    print("Session stopped; accepted artifacts remain available.")
                    return
                if action not in {"a", "accept", "approve"}:
                    print("Artifact was not approved; stopping at the current gate.")
                    return
                workflow.approve_artifact(specialist_artifact)
            workflow.accept(draft, automatic=True)
            continue
        while True:
            task_count = len(
                [task for task in workflow.session.tasks if task.stage == draft.stage]
            )
            task_states = ", ".join(
                f"{task.agent}:{task.state}"
                for task in workflow.session.tasks
                if task.stage == draft.stage
            )
            print(
                f"\n{'=' * 72}\nSTAGE: {draft.stage.upper()}  |  "
                f"SUPERVISOR BUNDLE: {task_count} SPECIALIST TASK(S)\n"
                f"TASK STATUS: {task_states or 'none'}\n"
                f"{'-' * 72}\n{draft.content}\n"
            )
            action = input("[a]ccept, [e]dit/revise, [s]top: ").strip().lower()
            if action in {"a", "accept"}:
                try:
                    workflow.accept(draft)
                    break
                except ValueError as exc:
                    print(f"Error: {exc}")
                    if workflow.session.status == "input_required":
                        fallback = (
                            input("[l]iterature-only fallback, [s]top: ")
                            .strip()
                            .lower()
                        )
                        if fallback in {"l", "literature", "literature-only"}:
                            workflow.accept_literature_only()
                            continue
                        workflow.stop()
                        return
                    continue
            if action in {"e", "edit", "revise"}:
                feedback = input("Your instructions for this stage: ").strip()
                try:
                    draft = workflow.revise(feedback)
                except ValueError as exc:
                    print(f"Error: {exc}")
                continue
            if action in {"s", "stop"}:
                workflow.stop()
                print("Session stopped; accepted stages remain available to save.")
                return
            print("Choose a, e, or s.")
    if workflow.done:
        print("\nAll stages accepted. The final report is ready to export.")
