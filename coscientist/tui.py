from __future__ import annotations

from .models import ApprovalProfile
from .orchestration import CoScientistWorkflow


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
    while not workflow.done and workflow.session.status == "active":
        workflow.advance_to_human_gate()
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
