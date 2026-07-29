"""Markdown research-dossier compiler with complete artifact provenance."""

from __future__ import annotations

import json
import re
from html import escape
from io import BytesIO
from pathlib import Path

from .models import (
    CandidatePopulation,
    DiscoveryManifest,
    DossierManifest,
    EvidencePacket,
    ResearchLandscape,
    ResearchPlan,
    ReviewSet,
    Session,
    TournamentState,
)


def _typed_summary(schema_name: str, payload: dict) -> list[str]:
    if not payload:
        return ["_No typed payload was available._"]
    if schema_name == "ResearchPlan":
        plan = ResearchPlan.model_validate(payload)
        return [
            f"- Intended claim: {plan.intended_claim}",
            f"- Research mode: {plan.research_mode}",
            f"- Constraints: {'; '.join(plan.constraints) or 'none recorded'}",
            f"- Stopping criteria: {'; '.join(plan.stopping_criteria)}",
        ]
    if schema_name == "EvidencePacket":
        packet = EvidencePacket.model_validate(payload)
        lines = [
            f"- Sources: {len(packet.sources)}",
            f"- Claim records: {len(packet.claims)}",
            f"- Packet verified: {'yes' if packet.verified else 'no'}",
        ]
        for source in packet.sources:
            lines.append(
                f"  - `{source.id}` — {source.verification_status}: {source.url}"
            )
        return lines + [f"- Limitation: {item}" for item in packet.limitations]
    if schema_name == "DiscoveryManifest":
        manifest = DiscoveryManifest.model_validate(payload)
        latest = manifest.coverage_history[-1] if manifest.coverage_history else None
        lines = [
            f"- Deep Research passes: {len(manifest.runs)}",
            f"- Distinct source leads: {len(manifest.source_leads)}",
            f"- Weighted coverage: {latest.weighted_score:.0%}"
            if latest
            else "- Weighted coverage: unavailable",
            f"- Convergence reason: {manifest.convergence_reason}",
            "- Verification status: discovered, not yet verified",
            "",
            "### Research directions",
        ]
        directions = list(
            dict.fromkeys(
                direction
                for narrative in manifest.narratives
                for direction in narrative.research_directions
            )
        )
        lines.extend(f"- {direction}" for direction in directions)
        lines.extend(["", "### Knowledge-base source leads"])
        lines.extend(
            f"- [{lead.title or 'Untitled source'}]({lead.canonical_url}) — "
            f"{lead.source_type.replace('_', ' ')}; discovered in pass "
            f"{', '.join(map(str, lead.originating_passes))}"
            for lead in manifest.source_leads
        )
        if latest and latest.gaps:
            lines.extend(["", "### Unresolved evidence gaps"])
            lines.extend(
                f"- **{gap.facet.replace('_', ' ')}:** {gap.description}"
                for gap in latest.gaps
            )
        return lines
    if schema_name == "CandidatePopulation":
        population = CandidatePopulation.model_validate(payload)
        lines = [
            f"Eight-candidate target: {population.target_size}; "
            f"actual candidates: {len(population.candidates)}."
        ]
        for index, candidate in enumerate(population.candidates, 1):
            lines.extend(
                [
                    "",
                    f"#### Candidate {index}",
                    "",
                    f"**Strategy:** {candidate.generation_strategy}",
                    "",
                    candidate.claim,
                    "",
                    f"**Rationale:** {candidate.rationale}",
                    "",
                    "**Predictions:**",
                    *[f"- {item}" for item in candidate.predictions],
                    "",
                    "**Competing explanations:**",
                    *[f"- {item}" for item in candidate.alternatives],
                    "",
                    f"**Falsifier:** {candidate.falsifier}",
                    "",
                    "**Risks:**",
                    *(
                        [f"- {item}" for item in candidate.risks]
                        or ["- Not yet characterized."]
                    ),
                    "",
                    "**Go/no-go tests:**",
                    *(
                        [f"- {item}" for item in candidate.go_no_go_tests]
                        or ["- Not yet specified."]
                    ),
                ]
            )
        if population.comparison_criteria:
            lines.extend(
                [
                    "",
                    "### Cross-candidate comparison criteria",
                    "",
                    *[f"- {criterion}" for criterion in population.comparison_criteria],
                ]
            )
        return lines
    if schema_name == "ReviewSet":
        reviews = ReviewSet.model_validate(payload).reviews
        lines = [
            "| Candidate | Criterion | Recommendation | Confidence | Fatal flaws |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for review in reviews:
            flaws = "; ".join(review.fatal_flaws) or "none recorded"
            lines.append(
                f"| `{review.candidate_id}` | {review.criterion} | "
                f"{review.recommendation} | {review.confidence:.2f} | {flaws} |"
            )
        return lines
    if schema_name == "TournamentState":
        tournament = TournamentState.model_validate(payload)
        lines = [
            "| Rank | Candidate | Elo |",
            "| ---: | --- | ---: |",
        ]
        for rank, (candidate_id, rating) in enumerate(
            sorted(tournament.ratings.items(), key=lambda item: -item[1]), 1
        ):
            lines.append(f"| {rank} | `{candidate_id}` | {rating:.1f} |")
        lines.extend(
            [
                "",
                f"- Pairwise comparisons: {len(tournament.comparisons)}",
                f"- Shortlist: {', '.join(f'`{item}`' for item in tournament.shortlist_ids)}",
                f"- Converged: {'yes' if tournament.converged else 'no'}",
            ]
        )
        return lines
    if schema_name == "EvolutionCycle":
        lines = []
        for record in payload.get("records", []):
            lines.extend(
                [
                    f"- `{record['id']}` from "
                    f"{', '.join(record.get('parent_ids', []))}: "
                    f"{'; '.join(record.get('changes', []))}",
                    f"  - New prediction: {record.get('new_prediction', '')}",
                    f"  - Mandatory re-review: {record.get('requires_rereview', True)}",
                ]
            )
        lines.extend(
            [
                f"- Re-reviews: {len(payload.get('rereviews', []))}",
                f"- Re-ranking rounds: {len(payload.get('ranking_history', []))}",
                f"- Converged: {payload.get('converged', False)}",
                f"- Stop reason: {payload.get('stop_reason', '')}",
            ]
        )
        return lines
    if schema_name == "ResearchLandscape":
        landscape = ResearchLandscape.model_validate(payload)
        lines = []
        for cluster in landscape.clusters:
            lines.append(
                f"- **{cluster.name}:** "
                f"{', '.join(f'`{item}`' for item in cluster.candidate_ids)}"
            )
        lines.extend(f"- Coverage gap: {item}" for item in landscape.coverage_gaps)
        return lines
    if schema_name == "DossierManifest":
        manifest = DossierManifest.model_validate(payload)
        return [
            f"- Recommended candidates: "
            f"{', '.join(f'`{item}`' for item in manifest.recommendation_candidate_ids) or 'none'}",
            f"- Candidates with unresolved fatal flaws: "
            f"{', '.join(f'`{item}`' for item in manifest.unresolved_fatal_flaw_candidate_ids) or 'none'}",
            *[
                f"- Evidence that would change the decision: {item}"
                for item in manifest.evidence_that_would_change_decision
            ],
        ]
    return [
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=False),
        "```",
    ]


def compile_dossier(session: Session) -> str:
    """Compile a concise front section followed by the complete audit appendix."""
    lines = [
        "# Co-Scientist Research Dossier",
        "",
        f"**Question:** {session.question}",
        f"**Research mode:** {session.research_mode}",
        f"**Approval profile:** {session.approval_profile}",
        f"**Approval policy:** {session.approval_mode}",
        f"**Research mode fallback:** "
        f"{'literature-only' if session.literature_only else 'full requested analysis'}",
        "",
        "## Research-integrity notice",
        "",
        "This dossier contains proposed hypotheses and review artifacts, not verified "
        "findings. A source satisfies an evidence gate only when its original content "
        "has been inspected and mapped to the exact claim. Auto approval is a workflow "
        "convenience and never constitutes scientific, safety, ethics, or institutional "
        "approval.",
        "",
        "## Executive synthesis",
        "",
    ]
    manifest_artifact = next(
        (
            artifact
            for artifact in reversed(session.artifacts)
            if artifact.schema_name == "DossierManifest"
        ),
        None,
    )
    if manifest_artifact:
        lines.extend(_typed_summary("DossierManifest", manifest_artifact.payload))
    else:
        lines.append("_The meta-review has not yet produced a dossier manifest._")

    if session.input_requirements:
        lines.extend(["", "## Input sufficiency", ""])
        for requirement in session.input_requirements:
            lines.append(
                f"- **{requirement.input_type}:** {requirement.status} — "
                f"{requirement.reason}"
            )

    section_order = (
        ("scope", "Research goal and constraints"),
        ("evidence", "Evidence landscape and verification"),
        ("generate", "Candidate generation"),
        ("reflect", "Independent reviews"),
        ("rank", "Tournament ranking"),
        ("evolve", "Candidate evolution"),
        ("proximity", "Research landscape"),
        ("meta_review", "Meta-review and decision conditions"),
    )
    for stage, title in section_order:
        artifacts = [
            artifact
            for artifact in session.artifacts
            if artifact.stage == stage and artifact.artifact_type == "specialist_output"
        ]
        if not artifacts:
            continue
        lines.extend(["", f"## {title}", ""])
        for artifact in artifacts:
            lines.extend(
                [
                    f"### {artifact.agent.replace('_', ' ').title()}",
                    "",
                    f"Artifact `{artifact.id}` · schema `{artifact.schema_name}` · "
                    f"status `{artifact.status}` · model `{artifact.producer_model}`",
                    "",
                    *_typed_summary(artifact.schema_name, artifact.payload),
                    "",
                ]
            )

    lines.extend(["", "## Complete artifact appendix", ""])
    for artifact in session.artifacts:
        lines.extend(
            [
                f"### `{artifact.id}` — {artifact.agent}",
                "",
                f"- Stage: `{artifact.stage}`",
                f"- Type: `{artifact.artifact_type}`",
                f"- Schema: `{artifact.schema_name}`",
                f"- Status: `{artifact.status}`",
                f"- Version: {artifact.version}",
                f"- Parent: `{artifact.parent_id or 'none'}`",
                f"- Inputs: "
                f"{', '.join(f'`{item}`' for item in artifact.input_artifact_ids) or 'none'}",
                f"- Checksum: `{artifact.checksum}`",
                "",
                artifact.content,
                "",
                "<details><summary>Validated typed payload</summary>",
                "",
                "```json",
                json.dumps(artifact.payload, indent=2, ensure_ascii=False),
                "```",
                "",
                "</details>",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision and task audit",
            "",
            "### Decisions",
            "",
            "| Time | Actor | Action | Stage | Artifact | Automatic |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for decision in session.decisions:
        lines.append(
            f"| {decision.created_at} | {decision.actor} | {decision.action} | "
            f"{decision.stage} | `{decision.artifact_id or ''}` | "
            f"{'yes' if decision.automatic else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### A2A/local task ledger",
            "",
            "| Task | Agent | Stage | State | Output |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for task in session.tasks:
        lines.append(
            f"| `{task.id}` | {task.agent} | {task.stage} | {task.state} | "
            f"`{task.output_artifact_id or ''}` |"
        )
    return "\n".join(lines)


def render_pdf(content: str) -> bytes:
    """Render a dossier as a downloadable PDF without touching the filesystem."""
    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("PDF export requires the reportlab dependency.") from exc

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "DossierBody",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        spaceAfter=3,
    )
    headings = {
        1: ParagraphStyle(
            "DossierH1",
            parent=body,
            fontSize=16,
            leading=20,
            spaceBefore=8,
            spaceAfter=8,
        ),
        2: ParagraphStyle(
            "DossierH2",
            parent=body,
            fontSize=13,
            leading=16,
            spaceBefore=7,
            spaceAfter=5,
        ),
        3: ParagraphStyle(
            "DossierH3",
            parent=body,
            fontSize=10,
            leading=13,
            spaceBefore=5,
            spaceAfter=3,
        ),
    }
    story = []
    in_code = False
    for line in content.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        level = len(line) - len(line.lstrip("#"))
        if level in headings and line[level : level + 1] == " ":
            story.append(Paragraph(escape(line[level + 1 :]), headings[level]))
        elif line == "---":
            story.append(PageBreak())
        elif not line:
            story.append(Spacer(1, 4))
        else:
            prefix = "• " if line.startswith("- ") else ""
            text = line[2:] if prefix else line
            if in_code:
                text = text.replace(" ", "&nbsp;")
            story.append(Paragraph(prefix + escape(text), body))
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=32,
        title="Co-Scientist Research Dossier",
    )
    document.build(story)
    return buffer.getvalue()


def _plain_markdown(value: str) -> str:
    """Keep dossier text readable in word processors without raw Markdown marks."""
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    return value.replace("**", "").replace("__", "").replace("`", "")


def render_docx(content: str) -> bytes:
    """Render an editable DOCX that can be opened directly in Google Docs."""
    try:
        from docx import Document
        from docx.enum.section import WD_SECTION
        from docx.shared import Inches, Pt
    except ImportError as exc:
        raise RuntimeError("DOCX export requires the python-docx dependency.") from exc

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    document.core_properties.title = "Co-Scientist Research Dossier"
    document.core_properties.subject = "Scientific research planning dossier"
    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9)

    in_code = False
    for line in content.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            document.add_heading(
                _plain_markdown(heading.group(2)),
                level=len(heading.group(1)),
            )
        elif line == "---":
            document.add_section(WD_SECTION.NEW_PAGE)
        elif line.startswith("- "):
            document.add_paragraph(
                _plain_markdown(line[2:]),
                style="List Bullet",
            )
        elif re.match(r"^\d+\.\s+", line):
            document.add_paragraph(
                _plain_markdown(re.sub(r"^\d+\.\s+", "", line)),
                style="List Number",
            )
        elif line.startswith("|") and line.endswith("|"):
            if set(line.replace("|", "").replace("-", "").replace(":", "").strip()):
                document.add_paragraph(
                    " | ".join(
                        _plain_markdown(cell.strip())
                        for cell in line.strip("|").split("|")
                    )
                )
        elif line:
            paragraph = document.add_paragraph(_plain_markdown(line))
            if in_code:
                paragraph.style = document.styles["No Spacing"]

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def write_dossier(path: str | Path, content: str) -> None:
    """Write Markdown, PDF, or Google Docs-compatible DOCX."""
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix == ".pdf":
        destination.write_bytes(render_pdf(content))
    elif suffix == ".docx":
        destination.write_bytes(render_docx(content))
    else:
        destination.write_text(content, encoding="utf-8")
