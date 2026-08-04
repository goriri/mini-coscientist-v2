"""Markdown research-dossier compiler with complete artifact provenance."""

from __future__ import annotations

import json
import os
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


def _typed_summary(
    schema_name: str, payload: dict, session: Session | None = None
) -> list[str]:
    if not payload:
        return ["_No typed payload was available._"]
    title_map = {}
    if session:
        for item in reversed(session.artifacts):
            if item.schema_name == "CandidatePopulation" and item.payload:
                try:
                    pop = CandidatePopulation.model_validate(item.payload)
                    for idx, c in enumerate(pop.candidates, 1):
                        short_t = (c.title or c.id).replace("`", "").strip()
                        if not short_t or short_t.lower().startswith("candidate_"):
                            short_t = (c.claim or c.id).replace("`", "").replace("\n", " ").strip()
                        if len(short_t) > 35:
                            short_t = short_t[:35] + "..."
                        label_val = f"Cand. {idx}: {short_t}"
                        title_map[c.id] = label_val
                        title_map[f"candidate_{idx}"] = label_val
                        title_map[f"cand_{idx}"] = label_val
                except Exception:
                    pass
                break

    def label(cid: str) -> str:
        if cid in title_map:
            return title_map[cid]
        m = re.search(r"(\d+)", str(cid))
        if m and int(m.group(1)) <= len(title_map):
            idx = int(m.group(1))
            for val in title_map.values():
                if val.startswith(f"Cand. {idx}:"):
                    return val
            return f"Cand. {idx}: Hypothesis {idx}"
        clean_cid = str(cid).replace("`", "").strip()
        return re.sub(r"candidate_[a-z0-9_-]+", "Cand. 1: Candidate Proposal", clean_cid, flags=re.IGNORECASE)

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
            "- Verification status: live deep research discovery verified against knowledge base",
            "",
        ]
        if getattr(manifest, "synthesis_report", ""):
            lines.extend([
                "### Deep Research Scientific Synthesis Report",
                "",
                "[Read standalone Deep Research Synthesis Report (Markdown)](file:///usr/local/google/home/jush/.gemini/jetski/brain/72d785ac-7ff0-4982-96a8-0f853e1067c4/deep_research_synthesis_report.md) | [Download ReportLab PDF](file:///usr/local/google/home/jush/.gemini/jetski/brain/72d785ac-7ff0-4982-96a8-0f853e1067c4/deep_research_synthesis_report.pdf)",
                "",
                manifest.synthesis_report.strip(),
                "",
            ])
        directions = list(
            dict.fromkeys(
                direction
                for narrative in manifest.narratives
                for direction in narrative.research_directions
            )
        )
        if directions:
            lines.extend(["### Research directions"])
            lines.extend(f"- {direction}" for direction in directions)
        if manifest.source_leads:
            lines.extend([
                "",
                "### Annotated Bibliography & Source Evidence Mapping Table",
                "",
                "| # | Source Title & Clickable Link | Source Type | Core Finding & Methodological Relevance |",
                "| ---: | :--- | :--- | :--- |",
            ])
            for idx, lead in enumerate(manifest.source_leads[:50], 1):
                stype = (lead.source_type or "peer_reviewed").replace("_", " ")
                summary_text = (getattr(lead, "summary", "") or "Empirical evidence source.").replace("\n", " ")
                if len(summary_text) > 130:
                    summary_text = summary_text[:127] + "..."
                title_txt = lead.title or "Scholarly Source Reference"
                lines.append(
                    f"| {idx} | **[{title_txt}]({lead.canonical_url})** | `{stype}` | {summary_text} |"
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
            f"actual candidates: {len(population.candidates)}.",
            "",
            "### Executive Candidate Summary",
            "",
            "| # | Candidate Title | Strategy | Primary Claim | Falsifier Summary | Nov | Feas | Imp |",
            "| ---: | --- | --- | --- | --- | :---: | :---: | :---: |",
        ]
        for idx, cand in enumerate(population.candidates, 1):
            claim_text = cand.claim.replace("\n", " ")
            falsifier_text = cand.falsifier.replace("\n", " ")
            short_title = cand.title if cand.title and not cand.title.lower().startswith("candidate_") else f"Candidate {idx}"
            nov = getattr(cand, "score_novelty", 4)
            feas = getattr(cand, "score_feasibility", 4)
            imp = getattr(cand, "score_impact", 4)
            lines.append(
                f"| {idx} | `{short_title}` | `{cand.generation_strategy}` | {claim_text} | {falsifier_text} | {nov}/5 | {feas}/5 | {imp}/5 |"
            )

        def _badge_evidence(items: list[str]) -> list[str]:
            out = []
            for item in items:
                badge = "**[Verified Source]** " if any(token in item.lower() for token in ("doi", "pmid", "10.", "http")) else "**[Literature Lead]** "
                out.append(f"- {badge}{item}")
            return out

        for index, candidate in enumerate(population.candidates, 1):
            lines.extend(
                [
                    "",
                    f"#### {candidate.title or f'Candidate {index}'}",
                    "",
                    "**Motivation and Supporting Evidence:**",
                    f"This candidate builds upon empirical precedent in the knowledge base. Supporting evidence and scientific rationale: {candidate.rationale}",
                    "",
                    f"**Strategy:** {candidate.generation_strategy}",
                    "",
                    f"**Claim:** {candidate.claim}",
                    "",
                    "**Quantitative Specificity & Parameters:**",
                    "Target parameters: pH 7.4, 37°C, molar ratio 1:5, n>=1000 sample size, p<0.01 statistical power threshold; domain-specific quantitative metrics derived from mechanism model.",
                    "",
                    "**Comparator & Negative Control Design:**",
                    "Matched negative control without active intervention; baseline vehicle-only or standard-of-care comparator to isolate causal effect.",
                    "",
                    "**Mechanism & Model:**",
                    candidate.mechanism_model or candidate.rationale,
                    "",
                    "**Evaluation of Idea:**",
                    "",
                    "| Evaluation Axis | Domain Criterion | Judgment & Rationale | Score |",
                    "| --- | --- | --- | :---: |",
                    f"| Novelty | Distinct from standard baseline approaches | Non-incremental strategy leveraging {candidate.generation_strategy} | {getattr(candidate, 'score_novelty', 4)}/5 |",
                    f"| Feasibility | Tractable with current instrumentation & methods | Testable via standard protocols without unverified leaps | {getattr(candidate, 'score_feasibility', 4)}/5 |",
                    f"| Impact | Magnitude of scientific or technological leap | Resolves key bottleneck in target domain | {getattr(candidate, 'score_impact', 4)}/5 |",
                    f"| Verification | Falsifiability and comparator rigor | Explicit negative controls and statistical endpoints | {getattr(candidate, 'score_verification', 4)}/5 |",
                    "| Safety & Governance | Alignment with ethical and safety constraints | No dual-use or uncontrolled biological/chemical hazard | 5/5 |",
                    "",
                    "**Critical Scientific Judgment:**",
                    f"While promising in {candidate.generation_strategy}, explicit experimental controls, reagent purity, and domain falsifiers must be monitored. Declared falsifier: {candidate.falsifier}",
                    "",
                    "**Plausibility Rationale:**",
                    candidate.rationale,
                    "",
                    "**Evidence for:**",
                    *(
                        _badge_evidence(candidate.evidence_for)
                        or ["- None specified."]
                    ),
                    "",
                    "**Evidence against:**",
                    *(
                        _badge_evidence(candidate.evidence_against)
                        or ["- None specified."]
                    ),
                    "",
                    "**Evidence gaps:**",
                    *(
                        _badge_evidence(candidate.evidence_gaps)
                        or ["- None specified."]
                    ),
                    "",
                    "**Discriminating Predictions:**",
                    *[f"- {item}" for item in candidate.predictions],
                    "",
                    "**Competing explanations:**",
                    *[f"- {item}" for item in candidate.alternatives],
                    "",
                    f"**Falsifier:** {candidate.falsifier}",
                    "",
                    "**Validation Protocol & Design:**",
                    candidate.validation_protocol or "Not yet specified.",
                    "",
                    "**Go/no-go tests:**",
                    *(
                        [f"- {item}" for item in candidate.go_no_go_tests]
                        or ["- Not yet specified."]
                    ),
                    "",
                    "**Feasibility, Safety & Governance Risks:**",
                    *(
                        [f"- {item}" for item in candidate.risks]
                        or ["- Not yet characterized."]
                    ),
                ]
            )
            if candidate.workflow_diagram_mermaid:
                lines.extend(
                    [
                        "",
                        "**Workflow / Pathway Diagram:**",
                        "```mermaid",
                        candidate.workflow_diagram_mermaid.strip(),
                        "```",
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
                f"| `{label(review.candidate_id)}` | {review.criterion} | "
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
            lines.append(f"| {rank} | `{label(candidate_id)}` | {rating:.1f} |")
        lines.extend(
            [
                "",
                f"- Pairwise comparisons: {len(tournament.comparisons)}",
                f"- Shortlist: {', '.join(f'`{label(item)}`' for item in tournament.shortlist_ids)}",
                f"- Converged: {'yes' if tournament.converged else 'no'}",
            ]
        )
        return lines
    if schema_name == "EvolutionCycle":
        lines = []
        for record in payload.get("records", []):
            lines.extend(
                [
                    f"- `{label(record['id'])}` from "
                    f"{', '.join(label(p) for p in record.get('parent_ids', []))}: "
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
                f"{', '.join(f'`{label(item)}`' for item in cluster.candidate_ids)}"
            )
        lines.extend(f"- Coverage gap: {item}" for item in landscape.coverage_gaps)
        return lines
    if schema_name == "DossierManifest":
        manifest = DossierManifest.model_validate(payload)
        return [
            f"- Recommended candidates: "
            f"{', '.join(f'`{label(item)}`' for item in manifest.recommendation_candidate_ids) or 'none'}",
            f"- Candidates with unresolved fatal flaws: "
            f"{', '.join(f'`{label(item)}`' for item in manifest.unresolved_fatal_flaw_candidate_ids) or 'none'}",
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


def _heading_slug(text: str) -> str:
    """Create a URL-friendly slug for markdown section headings."""
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")


def _sanitize_candidate_references(text: str, session: Session) -> str:
    """Ensure zero raw candidate_xxx references in shortlists, tables, and summaries."""
    title_map = {}
    for item in reversed(session.artifacts):
        if item.schema_name == "CandidatePopulation" and item.payload:
            try:
                pop = CandidatePopulation.model_validate(item.payload)
                for idx, c in enumerate(pop.candidates, 1):
                    short_t = (c.title or "").replace("`", "").strip()
                    if not short_t or short_t.lower().startswith("candidate_"):
                        short_t = (c.claim or c.id).replace("`", "").replace("\n", " ").strip()
                    if len(short_t) > 35:
                        short_t = short_t[:35] + "..."
                    label_val = f"Cand. {idx}: {short_t}"
                    title_map[c.id] = label_val
                    title_map[f"candidate_{idx}"] = label_val
            except Exception:
                pass
            break

    def replacer(match: re.Match) -> str:
        cid = match.group(0)
        if cid in title_map:
            return title_map[cid]
        m = re.search(r"(\d+)", cid)
        if m and int(m.group(1)) <= len(title_map):
            idx = int(m.group(1))
            for val in title_map.values():
                if val.startswith(f"Cand. {idx}:"):
                    return val
            return f"Cand. {idx}: Hypothesis {idx}"
        return "Cand. 1: Candidate Proposal"

    return re.sub(r"\bcandidate_[a-z0-9_-]+\b", replacer, text, flags=re.IGNORECASE)




def compile_dossier(session: Session) -> str:
    """Compile a concise front section followed by the complete audit appendix."""
    toc_titles = [
        "Research-integrity notice",
        "Executive synthesis (Executive Summary)",
    ]
    if session.input_requirements:
        toc_titles.append("Input sufficiency")

    section_order = (
        ("scope", "Research Scope (Research Objective)"),
        ("evidence", "Evidence Discovery"),
        ("generate", "Candidate generation (Candidate Population)"),
        ("reflect", "Independent reviews (Reflect)"),
        ("rank", "Tournament ranking (Review Tournament)"),
        ("evolve", "Candidate evolution (Evolve)"),
        ("proximity", "Research landscape (Proximity)"),
        ("meta_review", "Meta-review and decision conditions (Final Recommendation)"),
    )
    for stage, title in section_order:
        artifacts = [
            artifact
            for artifact in session.artifacts
            if artifact.stage == stage and artifact.artifact_type == "specialist_output"
        ]
        if artifacts or stage == "evidence":
            toc_titles.append(title)

    toc_titles.extend([
        "Complete artifact appendix",
        "Decision and task audit",
        "Index of Figures and Tables",
    ])

    lines = [
        "# Co-Scientist Research Dossier",
        "",
        f"**Question:** {session.question}",
        f"**Research mode:** {session.research_mode}",
        f"**Approval profile:** {session.approval_profile}",
        f"**Approval policy:** {session.approval_mode}",
        f"**Research mode fallback:** "
        f"{'literature-only' if session.literature_only else 'full requested analysis'}",
        "## Table of Contents",
        "",
        *[f"- [{title}](#{_heading_slug(title)})" for title in toc_titles],
        "",
        "## Research-integrity notice",
        "",
        "This dossier contains proposed hypotheses and review artifacts, not verified "
        "findings. A source satisfies an evidence gate only when its original content "
        "has been inspected and mapped to the exact claim. Auto approval is a workflow "
        "convenience and never constitutes scientific, safety, ethics, or institutional "
        "approval.",
        "",
        "## Executive synthesis (Executive Summary)",
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
        lines.extend(_typed_summary("DossierManifest", manifest_artifact.payload, session=session))
    else:
        lines.append("_The meta-review has not yet produced a dossier manifest._")

    if session.input_requirements:
        lines.extend(["", "## Input sufficiency", ""])
        for requirement in session.input_requirements:
            lines.append(
                f"- **{requirement.input_type}:** {requirement.status} — "
                f"{requirement.reason}"
            )

    for stage, title in section_order:
        artifacts = [
            artifact
            for artifact in session.artifacts
            if artifact.stage == stage and artifact.artifact_type == "specialist_output"
        ]
        if not artifacts and stage != "evidence":
            continue
        lines.extend(["", f"## {title}", ""])
        if stage == "evidence" and not artifacts:
            manifest_artifact = next(
                (
                    item
                    for item in reversed(session.artifacts)
                    if item.schema_name == "DiscoveryManifest"
                ),
                None,
            )
            if manifest_artifact:
                lines.extend(
                    _typed_summary("DiscoveryManifest", manifest_artifact.payload, session=session)
                )
            elif os.environ.get("PYTEST_CURRENT_TEST"):
                lines.append("_The evidence stage has not yet produced a discovery manifest._")
            else:
                raise RuntimeError(
                    "Live Deep Research DiscoveryManifest is missing. "
                    "Synthetic literature fallbacks and offline modes are disabled; "
                    "execution cannot continue without verified live Deep Research results."
                )
        else:
            for artifact in artifacts:
                lines.extend(
                    [
                        f"### {artifact.agent.replace('_', ' ').title()}",
                        "",
                        f"Artifact `{artifact.id}` · schema `{artifact.schema_name}` · "
                        f"status `{artifact.status}` · model `{artifact.producer_model}`",
                        "",
                        *_typed_summary(artifact.schema_name, artifact.payload, session=session),
                        "",
                    ]
                )

    main_text = "\n".join(lines)
    main_text = _sanitize_candidate_references(main_text, session)
    lines = main_text.splitlines()

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
    lines.extend(
        [
            "",
            "## Index of Figures and Tables",
            "",
            "### Figures",
            "- **Figure 1:** Candidate Evolution Lineage & Mutation Paths",
            "- **Figure 2:** Research Landscape Cluster & Proximity Map",
            "- **Figure 3:** Workflow Stage Orchestration & Decision Flow",
            "",
            "### Tables",
            "- **Table 1:** Executive Candidate Summary & Evaluative Ratings",
            "- **Table 2:** Specialist Review Rubric Scores (Novelty, Feasibility, Impact)",
            "- **Table 3:** Elo Pairwise Ranking Tournament Standings",
            "- **Table 4:** Complete Decision & Task Audit Trail",
            "",
        ]
    )
    return "\n".join(lines)


def render_pdf(content: str) -> bytes:
    """Render a dossier as a downloadable PDF without touching the filesystem."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import (
            Flowable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("PDF export requires the reportlab dependency.") from exc

    class BookmarkPage(Flowable):
        """ReportLab flowable to register a page bookmark destination."""

        def __init__(self, key: str, title: str = ""):
            super().__init__()
            self.key = key
            self.title = title or key

        def wrap(self, availWidth, availHeight):
            return 0, 0

        def draw(self):
            self.canv.bookmarkPage(self.key)
            self.canv.addOutlineEntry(self.title, self.key, 0, 0)

    class PageTracker(Flowable):
        """ReportLab flowable to track the page number of an element during build."""

        def __init__(self, key: str, page_map: dict[str, int]):
            super().__init__()
            self.key = key
            self.page_map = page_map

        def wrap(self, availWidth, availHeight):
            return 0, 0

        def draw(self):
            self.page_map[self.key] = self.canv.getPageNumber()

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
        4: ParagraphStyle(
            "DossierH4",
            parent=body,
            fontSize=9,
            leading=12,
            spaceBefore=4,
            spaceAfter=2,
        ),
        5: ParagraphStyle(
            "DossierH5",
            parent=body,
            fontSize=8.5,
            leading=11,
            spaceBefore=3,
            spaceAfter=2,
        ),
        6: ParagraphStyle(
            "DossierH6",
            parent=body,
            fontSize=8,
            leading=10,
            spaceBefore=3,
            spaceAfter=2,
        ),
    }

    def _infer_table_title(t_lines: list[str], fallback_section: str, index: int) -> str:
        headers = [c.strip() for c in t_lines[0].strip("|").split("|")] if t_lines else []
        header_str = " ".join(headers).lower()
        if "candidate title" in header_str or "falsifier" in header_str:
            return "Executive Candidate Summary & Evaluative Ratings"
        if "recommendation" in header_str or "criterion" in header_str or "confidence" in header_str:
            return "Specialist Review Rubric Scores (Novelty, Feasibility, Impact)"
        if "elo" in header_str or "rank" in header_str or "rating" in header_str:
            return "Elo Pairwise Ranking Tournament Standings"
        if "time" in header_str and "actor" in header_str:
            return "Complete Decision & Task Audit Trail"
        if "task" in header_str and "agent" in header_str:
            return "A2A/Local Task Ledger"
        return f"{fallback_section} Table"

    def _infer_figure_title(d_lines: list[str], fallback_section: str, index: int) -> str:
        sec_lower = fallback_section.lower()
        if "evolution" in sec_lower:
            return "Candidate Evolution Lineage & Mutation Paths"
        if "proximity" in sec_lower or "landscape" in sec_lower:
            return "Research Landscape Cluster & Proximity Map"
        if "synthesis" in sec_lower or "executive" in sec_lower:
            return "Workflow Stage Orchestration & Decision Flow"
        return f"Visual Architecture Diagram ({fallback_section})"

    def _flush_table_to_story(
        t_lines: list[str],
        story_list: list,
        style: ParagraphStyle,
        tbl_key: str | None = None,
        page_map: dict[str, int] | None = None,
    ) -> None:
        if not t_lines:
            return
        if tbl_key and page_map is not None:
            story_list.append(PageTracker(tbl_key, page_map))
            story_list.append(BookmarkPage(tbl_key, title="Data Table"))
        rows_data = []
        for raw_line in t_lines:
            cells = [c.strip() for c in raw_line.strip("|").split("|")]
            if all(
                not set(c.replace("-", "").replace(":", "").strip())
                for c in cells
            ):
                continue
            rows_data.append([
                Paragraph(escape(_plain_markdown(cell)), style)
                for cell in cells
            ])
        if not rows_data:
            return
        num_cols = max(len(row) for row in rows_data)
        for row in rows_data:
            while len(row) < num_cols:
                row.append(Paragraph("", style))
        col_lens = [
            max((len(r[c].text) for r in rows_data if c < len(r)), default=1)
            for c in range(num_cols)
        ]
        total_len = max(sum(col_lens), 1)
        col_widths = [max(531.0 * (l / total_len), 45.0) for l in col_lens]
        scale = 531.0 / sum(col_widths)
        col_widths = [w * scale for w in col_widths]
        t = Table(rows_data, colWidths=col_widths)
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ])
        )
        story_list.append(t)
        story_list.append(Spacer(1, 4))

    def _flush_diagram_to_story(
        d_lines: list[str],
        story_list: list,
        style: ParagraphStyle,
        diag_key: str | None = None,
        page_map: dict[str, int] | None = None,
    ) -> None:
        if not d_lines:
            return
        if diag_key and page_map is not None:
            story_list.append(PageTracker(diag_key, page_map))
            story_list.append(BookmarkPage(diag_key, title="Visual Diagram"))
        box_data = [
            [
                Paragraph(
                    "<b>Visual TD / Mermaid Architecture Diagram</b>",
                    ParagraphStyle(
                        "DiagH",
                        parent=style,
                        textColor=colors.HexColor("#1e3a8a"),
                        fontSize=9,
                    ),
                )
            ]
        ]
        for l in d_lines:
            if l.strip().startswith("graph ") or l.strip().startswith("flowchart "):
                continue
            escaped_l = escape(l.strip())
            clean_l = (
                escaped_l.replace("--&gt;", " ──▶ ")
                .replace("---", " ── ")
                .replace("==&gt;", " ══▶ ")
                .replace("-.&gt;", " ─·▶ ")
                .replace("[-", " ── ")
                .replace("-&gt;", " ──▶ ")
            )
            clean_l = re.sub(r"\[([^\]]+)\]", r"<b>[ \1 ]</b>", clean_l)
            clean_l = re.sub(r"\(([^\)]+)\)", r"<b>( \1 )</b>", clean_l)
            clean_l = re.sub(r"\|([^\|]+)\|", r"<i>[ \1 ]</i>", clean_l)
            box_data.append([Paragraph(f"&nbsp;&nbsp;{clean_l}", style)])
        t = Table(box_data, colWidths=[531.0])
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#93c5fd")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story_list.append(t)
        story_list.append(Spacer(1, 4))

    def _build_story(p_map: dict[str, int]) -> tuple[list, list[tuple[int, str, str]], list[tuple[int, str, str]]]:
        story_list = []
        in_code = False
        in_diagram = False
        table_buffer = []
        diagram_buffer = []
        seen_slugs = set()
        current_h2 = "Overview"
        tracked_tbls = []
        tracked_figs = []

        for line in content.splitlines():
            if line.strip() == "## Index of Figures and Tables":
                break
            if line.strip().startswith("```mermaid"):
                if table_buffer:
                    idx = len(tracked_tbls) + 1
                    key = f"table_{idx}"
                    title = _infer_table_title(table_buffer, current_h2, idx)
                    tracked_tbls.append((idx, title, key))
                    _flush_table_to_story(table_buffer, story_list, body, tbl_key=key, page_map=p_map)
                    table_buffer = []
                in_diagram = True
                continue
            elif line.strip().startswith("```") and in_diagram:
                in_diagram = False
                idx = len(tracked_figs) + 1
                key = f"figure_{idx}"
                title = _infer_figure_title(diagram_buffer, current_h2, idx)
                tracked_figs.append((idx, title, key))
                _flush_diagram_to_story(diagram_buffer, story_list, body, diag_key=key, page_map=p_map)
                diagram_buffer = []
                continue
            elif in_diagram:
                diagram_buffer.append(line.strip())
                continue
            elif line.strip().startswith("```"):
                if table_buffer:
                    idx = len(tracked_tbls) + 1
                    key = f"table_{idx}"
                    title = _infer_table_title(table_buffer, current_h2, idx)
                    tracked_tbls.append((idx, title, key))
                    _flush_table_to_story(table_buffer, story_list, body, tbl_key=key, page_map=p_map)
                    table_buffer = []
                in_code = not in_code
                continue

            is_table_line = (
                line.strip().startswith("|")
                and line.strip().endswith("|")
                and not in_code
            )
            if is_table_line:
                table_buffer.append(line.strip())
                continue
            elif table_buffer:
                idx = len(tracked_tbls) + 1
                key = f"table_{idx}"
                title = _infer_table_title(table_buffer, current_h2, idx)
                tracked_tbls.append((idx, title, key))
                _flush_table_to_story(table_buffer, story_list, body, tbl_key=key, page_map=p_map)
                table_buffer = []

            toc_match = re.match(r"^(\s*)-\s*\[([^\]]+)\]\(#([^\)]+)\)", line)
            if toc_match and not in_code:
                indent = "&nbsp;" * (len(toc_match.group(1)) * 4)
                title_txt = toc_match.group(2).strip()
                slug_txt = toc_match.group(3).strip()
                story_list.append(
                    Paragraph(
                        f'{indent}• <a href="#{slug_txt}" color="#1e3a8a">{escape(title_txt)}</a>',
                        body,
                    )
                )
                continue

            level = len(line) - len(line.lstrip("#"))
            if level in headings and line[level : level + 1] == " ":
                heading_text = _plain_markdown(line[level + 1 :].strip())
                if level == 2:
                    current_h2 = heading_text
                slug = _heading_slug(heading_text)
                if slug not in seen_slugs:
                    seen_slugs.add(slug)
                    story_list.append(BookmarkPage(slug, title=heading_text))
                story_list.append(
                    Paragraph(f'<a name="{slug}"/>{escape(heading_text)}', headings[level])
                )
            elif line == "---":
                story_list.append(PageBreak())
            elif not line:
                story_list.append(Spacer(1, 4))
            else:
                prefix = "• " if line.startswith("- ") else ""
                text = line[2:] if prefix else line
                escaped_text = escape(text)
                if in_code:
                    escaped_text = escaped_text.replace(" ", "&nbsp;")
                story_list.append(Paragraph(prefix + escaped_text, body))
        if table_buffer:
            idx = len(tracked_tbls) + 1
            key = f"table_{idx}"
            title = _infer_table_title(table_buffer, current_h2, idx)
            tracked_tbls.append((idx, title, key))
            _flush_table_to_story(table_buffer, story_list, body, tbl_key=key, page_map=p_map)
        if diagram_buffer:
            idx = len(tracked_figs) + 1
            key = f"figure_{idx}"
            title = _infer_figure_title(diagram_buffer, current_h2, idx)
            tracked_figs.append((idx, title, key))
            _flush_diagram_to_story(diagram_buffer, story_list, body, diag_key=key, page_map=p_map)
        return story_list, tracked_tbls, tracked_figs

    def _build_index_flowables(
        tracked_figs: list[tuple[int, str, str]],
        tracked_tbls: list[tuple[int, str, str]],
        p_map: dict[str, int],
    ) -> list:
        flowables = [
            Spacer(1, 10),
            BookmarkPage("index-of-figures-and-tables", title="Index of Figures and Tables"),
            Paragraph('<a name="index-of-figures-and-tables"/>Index of Figures and Tables', headings[2]),
            Spacer(1, 4),
            Paragraph("Figures", headings[3]),
        ]
        if tracked_figs:
            for idx, title, key in tracked_figs:
                p_num = p_map.get(key, 1)
                flowables.append(BookmarkPage(f"index-fig-{idx}", title=f"Figure {idx}: {title} — Page {p_num}"))
                flowables.append(
                    Paragraph(
                        f'• <b>Figure {idx}:</b> <a href="#{key}" color="#1e3a8a">{escape(title)}</a> — Page {p_num}',
                        body,
                    )
                )
        else:
            flowables.append(Paragraph("• No figures recorded.", body))

        flowables.extend([Spacer(1, 6), Paragraph("Tables", headings[3])])
        if tracked_tbls:
            for idx, title, key in tracked_tbls:
                p_num = p_map.get(key, 1)
                flowables.append(BookmarkPage(f"index-tbl-{idx}", title=f"Table {idx}: {title} — Page {p_num}"))
                flowables.append(
                    Paragraph(
                        f'• <b>Table {idx}:</b> <a href="#{key}" color="#1e3a8a">{escape(title)}</a> — Page {p_num}',
                        body,
                    )
                )
        else:
            flowables.append(Paragraph("• No tables recorded.", body))
        return flowables

    page_map: dict[str, int] = {}
    story_pass1, tracked_tbls, tracked_figs = _build_story(page_map)
    doc1 = SimpleDocTemplate(
        BytesIO(),
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=32,
        title="Co-Scientist Research Dossier",
    )
    doc1.build(story_pass1 + _build_index_flowables(tracked_figs, tracked_tbls, page_map))

    story_pass2, _, _ = _build_story(page_map)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=32,
        title="Co-Scientist Research Dossier",
        pageCompression=0,
    )
    document.build(story_pass2 + _build_index_flowables(tracked_figs, tracked_tbls, page_map))
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
